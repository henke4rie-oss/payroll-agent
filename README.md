# payroll-agent

A small AI agent that answers payroll questions for a South African small business. It looks up employees, calculates PAYE / UIF / net pay using the SARS 2026/27 tax tables, explains the tax behind a payslip, and can add, edit, rename and remove employees. Every change needs a human to approve it first and is written to an audit log.

It sits on top of the tax engine from my [PayrollZA](https://github.com/henke4rie-oss/PayrollZA) project (`payroll_agent/engine.py` is a copy of that engine).

## Why an agent and not a workflow?

A fixed workflow or form is the right tool for a routine payroll run. An agent earns its place at the front end, where requests are loose and vary: "make Sipho's salary 25k", "what would Thandi take home at 66?", "who's our biggest cost?", "fix the spelling of Zanele's surname". Each needs a different sequence of tools, and names can be vague or mistyped. The model chooses the next step, but my code controls what it is allowed to do. The money calculations stay in plain deterministic code.

## How it works

1. The user types a request
2. The model turns it into a tool call with structured arguments: which person, which figures (`fields`), which period, or which team-wide answer (`team_pay`)
3. My code runs the tool, which uses the deterministic engine for every number
4. For clear read-only results, my code writes the reply from the tool's arguments and numbers, with no second model call; for changes, errors and "why" questions, the result goes back to the model
5. This repeats until there is an answer (maximum 6 steps)

The model decides *what was asked* (it is good with language: "thandis uif", "tax numbers", "who takes home the least"), and code decides *what is true* (the numbers, the wording of figures, the approval check). There is no phrase-matching in code to maintain, so a new phrasing does not need a new rule.

Salaries can be quoted per month, year, week or day, and the answer can be shown in a different period. A daily rate without a work schedule makes the tool return an error, so the model asks for the number of paid workdays per week.

### Evaluation

`python -m evals.run` runs 30 real phrasings (typos, possessives, follow-ups, rankings, what-ifs, declined changes) against the real model and grades each reply for the right figures, the wrong figures that must not appear, and invented amounts. Results are in `evals/results/` and the first baseline is in `evals/baseline.json`. It uses real tokens (about 30k per run), so it is run on demand, not as part of `pytest`.

### Tools

| Tool | What it does | Changes data? |
|---|---|---|
| `employee_pay` | One employee (or a given salary and age): chosen figures, what-ifs, any period, or the tax breakdown. Suggests close names on a typo | No |
| `team_pay` | Everyone: list, count, who has the highest or lowest of a figure, company totals | No |
| `change_employee` | `approve` a payslip, `update` salary or age, `add`, `rename` (full current name) or `remove` (full name; a copy goes in the audit log) | Yes, needs approval |

Under these three sit the small functions in `tools.py` (lookup, payslip, update, add, rename, remove). They are what the tests exercise directly and what keeps each rule in one place.

## Design decisions

- **The model never does maths.** All money calculations use exact `Decimal` arithmetic in the engine.
- **Numbers come from the engine, not the model.** The final payslip figures are calculated and attached by code.
- **Human-in-the-loop on every change.** The approval check is in the tool code, so the model cannot skip it. The person sees a before and after (including the net pay effect) and nothing is saved unless they type `y`.
- **Tools take names and values only.** Payslips and totals are recalculated in code, so the model can't get an invented number approved.
- **Risky actions need the exact full name.** Rename and remove refuse partial names, so "Sipho" can never delete the wrong Sipho.
- **Typos are caught, not guessed.** A not-found error suggests close names (difflib) and the prompt tells the model to ask the user which one they meant.
- **Input validation.** Names are tidied (extra spaces) and limited to letters, spaces, apostrophes, hyphens and full stops. Duplicate names are refused, ignoring case. Ages must be whole numbers and the engine rejects impossible ages. Salaries must be above zero.
- **"What if" is not a change.** Questions like "what would her pay be at 66?" use the calculation tool and never touch the data.
- **Audit log.** Every approved change is recorded with a timestamp.
- **Step limit and retries.** A confused model can't loop forever, and temporary API errors (rate limits, overload) are retried with a growing wait.
- **Layers are separate files.** `engine.py` (tax maths), `store.py` (data), `tools.py` (what the agent can do), `agent.py` (the loop), `llm.py` (the only file that knows about the model provider). Tests use a scripted fake model, so they need no API key.

## Setup

1. `python -m venv .venv` and activate it (Windows: `.venv\Scripts\activate`)
2. `pip install -r requirements.txt`
3. Get a free API key from console.groq.com
4. Copy `.env.example` to `.env` and set `LLM_API_KEY` (optionally set `LLM_MODEL`; defaults to `openai/gpt-oss-20b`)
5. `python main.py` for clean output, or `python main.py --stats` to see each model call, tool call, timing and token count.

No Docker needed: this project has no database or server.

Things to try:
- `What is Thandi Nkosi's net pay?`
- `What would Thandi's pay be at 66?` (what-if, changes nothing)
- `Change Thandi's age to 66` (asks you to approve)
- `Why is Johan van der Merwe's tax lower than you'd expect?`
- `Sipho's salary is now 30k` (two Siphos, so it should ask which one)
- `Change Thandy's age to 66` (typo, so it should suggest Thandi)
- `Fix the spelling of Thandi Nkosi to Thandi Nkosie`
- `Add Zanele Mkhize, 35, R30000`
- `What's the total payroll?`

## Tests

```bash
python -m pytest
```

The tests cover the tax calculations (expected values worked out by hand from the SARS table), every tool, the approval gates, typo and validation edge cases, the step limit, unknown tool requests and the retry logic. They use a fake model, so they do not test live model behaviour.

## Limitations

- Conversation history is retained for the current CLI session (up to 10 exchanges); restarting clears it
- Employees are a small fake JSON file, and approved edits are written straight to it (no database, no concurrent-edit protection)
- Age is editable for demo purposes. A real system would derive age from the ID number or birthdate.
- PAYE is a simple monthly annualised calculation (no pension, medical credits or ETI)
- No authentication or roles: whoever runs the program can approve
- Behaviour with the live model depends on the model version and prompt. Tests do not cover this yet.

## What would break first at scale

1. The JSON file store (use Postgres, as in PayrollZA)
2. Single-user sessions (add per-user sessions and roles for approvals)
3. LLM cost and rate limits (cache, use a cheaper model for simple requests, queue requests)

## Roadmap

- Conversation memory for follow-up questions
- An evaluation script that runs real questions against the live model and checks the tools it chose
- Inventory tracking and invoicing tools behind a router agent

## AI tools used

> EDIT THIS BEFORE SUBMITTING: say which AI tools you used and what you personally changed, ran and checked.

This project was built with help from Claude (Anthropic). I reviewed the code and tests and can explain how each part works.
