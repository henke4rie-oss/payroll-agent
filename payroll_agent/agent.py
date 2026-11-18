"""The agent loop.

An agent is a loop:
    1. Show the model the conversation so far
    2. If it asks for a tool, run the tool and add the result to the conversation
    3. Repeat until the model answers without asking for a tool (or we hit MAX_STEPS)

A fixed workflow decides the steps in advance. Here the MODEL decides which tool to use
next, and which figures the user asked for (as tool arguments). Our code decides what is
allowed (the tool list, the step limit, the human approval) and writes the exact numbers.
"""

import re
import time
from typing import Callable, Optional

from .llm import ModelTurn, ToolCall, ToolResults, UserMessage
from .schemas import AgentResult
from .tools import READ_ONLY_TOOLS, Toolbox

MAX_STEPS = 6  # safety limit so a confused model can't loop forever (and burn your free quota)
MAX_CONVERSATION_MESSAGES = 8

SYSTEM_PROMPT = """Payroll assistant for a small South African business. Amounts in Rands.
- Every number comes from a tool; never calculate yourself.
- Call a tool first; ask the user only if the tool reports a problem. Never ask for data the tools hold (ages, salaries). A name can be partial or possessive ("sipho", "thandis"): pass it as typed and the tool resolves it.
- One person: employee_pay. Everyone (counts, who earns or pays the most, totals, filters like "over 60"): team_pay. When the user clearly asks to change, approve, add, rename or remove, call change_employee directly; a person confirms inside the tool.
- Set fields to exactly the figures asked ("uif" -> ["uif"]). Leave it empty for a full payslip, "tax numbers" or "everything". Use explain only for "why" questions. Tax means PAYE. SDL is paid by the employer.
- What-ifs: employee_pay with the name plus the changed age or salary. If a salary is quoted per year, week or day, pass salary_period; use period for how the answer should be shown.
- Follow-ups ("and sipho?", "what about 66?") keep the earlier person and figures unless the user changes them.
- If a tool says a name is unknown or ambiguous, or a value is missing (including paid days per week for a daily rate), ask one short question.
- Money as R25,000.00. Keep replies short and plain."""

# Words that suggest the user wants to change data, so the change_employee tool must be offered.
# Hiding it is only a token saving: if this misses, the model says it can't, and approval is enforced in the tool.
WRITE_INTENT = re.compile(
    r"\b(?:approv\w*|finali[sz]\w*|updat\w*|chang\w*|set|rais\w*|increas\w*|decreas\w*|add\w*|hir\w*|"
    r"renam\w*|spell\w*|misspel\w*|typo|correct\w*|fix\w*|remov\w*|delet\w*|fire[ds]?|left|leav\w*)\b",
    re.IGNORECASE,
)

LABELS = {"gross": "gross pay", "paye": "PAYE", "uif": "UIF", "sdl": "SDL (paid by the employer)", "net": "net pay"}
PERIOD_PHRASE = {"month": "a month", "year": "a year", "week": "a week, averaged over 52 weeks"}
SALARY_PHRASE = {"month": "a month", "year": "a year", "week": "a week", "day": "a day"}
PERIOD_LABEL = {"month": "monthly", "year": "annual", "week": "weekly average (52 weeks per year)"}


def run_tool(toolbox: Toolbox, call: ToolCall) -> dict:
    """Run one tool the model asked for. Never crashes: problems go back to the model as an error."""
    function = toolbox.model_tools.get(call.name)
    if function is None:
        return {"error": f"Unknown tool '{call.name}'. Available tools: {', '.join(toolbox.model_tools)}"}
    try:
        return function(**call.args)
    except TypeError as err:  # model sent missing or extra arguments
        return {"error": f"Bad arguments for {call.name}: {err}"}


def _tools_for(question: str, conversation_history: Optional[list]) -> Optional[list[str]]:
    """Offer the write tool only when this or the previous user message sounds like a change."""
    previous = [m.text for m in (conversation_history or []) if isinstance(m, UserMessage)][-1:]
    if any(WRITE_INTENT.search(text) for text in (question, *previous)):
        return None
    return list(READ_ONLY_TOOLS)


def _money(value: float) -> str:
    return f"R{value:,.2f}"


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


def _period_phrase(result: dict) -> str:
    if result["period"] == "day":
        return f"per paid workday, at {result['days_per_week']} paid days per week"
    return PERIOD_PHRASE[result["period"]]


def _full_employee_reply(result: dict) -> str:
    name = result.get("name")
    who = f"{name} at age {result['age']}" if name else f"age {result['age']}"
    note = " (what-if, nothing was changed)" if name and result.get("what_if") else ""
    if result["period"] == "month":
        return (f"For {who} on {_money(result['gross'])} gross per month{note}: PAYE {_money(result['paye'])}, "
                f"UIF {_money(result['uif'])}, net pay {_money(result['net'])}.")
    label = (f"per paid workday ({result['days_per_week']} paid days per week)"
             if result["period"] == "day" else PERIOD_LABEL[result["period"]])
    return (f"For {who}, estimated {label} gross pay is {_money(result['gross'])}, "
            f"PAYE is {_money(result['paye'])}, employee UIF is {_money(result['uif'])} "
            f"and net pay is {_money(result['net'])}{note}.")


def _employee_reply(result: dict) -> str:
    name = result.get("name")
    fields = result["fields"]
    if fields == ["age"] and name:
        return f"{name} is {result['age']}."
    figures = [f for f in fields if f in LABELS]
    if not figures:
        return _full_employee_reply(result)

    what_if = result.get("what_if", False)
    verb = "would be" if what_if or not name else "is"
    body = _join([f"{LABELS[f]} {verb} {_money(result[f])}" for f in figures])
    phrase = _period_phrase(result)
    if not name:
        inputs = f"age {result['age']}, {_money(result['salary'])} gross {SALARY_PHRASE[result['salary_period']]}"
        return _capitalise(f"{body} {phrase} ({inputs}).")

    context = []
    if result.get("age_override"):
        context.append(f"at age {result['age']}")
    if "salary" in result:
        context.append(f"on {_money(result['salary'])} gross {SALARY_PHRASE[result['salary_period']]}")
    prefix = _capitalise(" ".join(context)) + ", " if context else ""
    note = " (nothing was changed)" if what_if else ""
    return f"{prefix}{name}'s {body} {phrase}{note}."


def _ranking_reply(names: list[str], figure: str, value: float, highest: bool) -> str:
    many = len(names) > 1
    who = _join(names) + (" both" if len(names) == 2 else " all" if many else "")
    if figure == "age":
        return f"{who} {'are' if many else 'is'} the {'oldest' if highest else 'youngest'}: {int(value)}."
    extreme = "most" if highest else "least"
    amount = _money(value)
    if figure == "gross":
        return f"{who} {'earn' if many else 'earns'} the {extreme}: {amount} a month."
    if figure == "net":
        return f"{who} {'take' if many else 'takes'} home the {extreme}: {amount} a month."
    if figure == "sdl":
        return (f"{who} {'have' if many else 'has'} the {'highest' if highest else 'lowest'} "
                f"{LABELS[figure]}: {amount} a month.")
    return f"{who} {'pay' if many else 'pays'} the {extreme} {LABELS[figure]}: {amount} a month."


def _team_reply(result: dict) -> Optional[str]:
    answer = result["answer"]
    if answer == "count":
        count = result["count"]
        return f"There {'is' if count == 1 else 'are'} {count} employee{'' if count == 1 else 's'}."
    if answer in ("highest", "lowest"):
        return _ranking_reply(result["names"], result["figure"], result["value"], answer == "highest")
    if answer == "total":
        totals, count = result["totals"], result["count"]
        people = f"{count} employee{'' if count == 1 else 's'}"
        if result["figure"] == "all":
            return (f"Monthly payroll for {people}: gross {_money(totals['gross'])}, PAYE {_money(totals['paye'])}, "
                    f"employee UIF {_money(totals['uif'])}, net pay {_money(totals['net'])}.")
        figure = result["figure"]
        return f"For {people}, total monthly {LABELS[figure]} is {_money(totals[figure])}."
    return None  # "list": the model reads the data itself (filters, "list everyone")


def _format_reply(tool_name: str, args: dict, result: dict) -> Optional[str]:
    """Put a read-only tool result into words without another model call. None = let the model answer.
    This looks only at what the model asked for (the tool arguments) and what the tool returned."""
    if "error" in result:
        return None
    if tool_name == "employee_pay" and not args.get("explain"):
        return _employee_reply(result)
    if tool_name == "team_pay":
        return _team_reply(result)
    return None


def _change_reply(args: dict, result: dict, toolbox: Toolbox) -> Optional[str]:
    """Words for an approved or declined change, written by code. None = let the model answer (errors)."""
    status = result.get("status")
    if status == "declined":
        return "Cancelled. Nothing was changed."
    if status != "approved":
        return None
    payslip = toolbox.state.last_payslip
    net = f" New net pay: {_money(payslip.net_pay)} a month." if payslip else ""
    action = args.get("action")
    if action == "update":
        old, new = result["old_value"], result["new_value"]
        if result["field"] == "gross_monthly":
            return f"Updated {result['employee']}'s salary from {_money(old)} to {_money(new)}.{net}"
        notes = []
        if old < 65 <= new:
            notes.append("age 65+ adds the secondary rebate")
        if old < 75 <= new:
            notes.append("age 75+ adds the tertiary rebate")
        extra = f" {_capitalise(_join(notes))}." if notes else ""
        return f"Updated {result['employee']}'s age from {old} to {new}.{net}{extra}"
    if action == "add":
        return f"Added {result['employee']}."
    if action == "rename":
        return f"Renamed {result['old_name']} to {result['new_name']}."
    if action == "remove":
        return f"Removed {result['employee']}. Their details are kept in the audit log."
    if action == "approve":
        figure = f": net pay {_money(payslip.net_pay)} a month" if payslip else ""
        return f"Approved {result['employee']}'s payslip{figure}."
    return None


def _remember_conversation(
    conversation_history: Optional[list],
    question: str,
    answer: str,
) -> None:
    if conversation_history is not None:
        conversation_history.extend((
            UserMessage(question),
            ModelTurn(text=answer, raw={"role": "assistant", "content": answer}),
        ))
        del conversation_history[:-MAX_CONVERSATION_MESSAGES]


def run_agent(
    question: str,
    llm,
    toolbox: Toolbox,
    log: Callable[[str], None] = lambda message: None,
    conversation_history: Optional[list] = None,
) -> AgentResult:
    toolbox.state.reset()
    history: list = list(conversation_history or [])
    history.append(UserMessage(question))

    tool_names = _tools_for(question, conversation_history)
    model_seconds = 0.0
    tool_seconds = 0.0
    model_calls = 0
    tokens_in = tokens_out = 0

    def log_timing() -> None:
        log(f"timing: {model_calls} model calls took {model_seconds:.3f}s, tools took {tool_seconds:.3f}s, "
            f"tokens in={tokens_in} out={tokens_out}")

    for step in range(1, MAX_STEPS + 1):
        log(f"step {step}: asking the model...")
        started = time.perf_counter()
        turn: ModelTurn = llm.step(history, tool_names=tool_names)
        elapsed = time.perf_counter() - started
        model_seconds += elapsed
        model_calls += 1
        if turn.usage:
            tokens_in += turn.usage[0]
            tokens_out += turn.usage[1]
        log(f"step {step}: model responded in {elapsed:.3f}s")
        history.append(turn)

        if not turn.tool_calls:
            break  # the model is done asking for tools

        results = []
        call_ids = []
        for call in turn.tool_calls:
            log(f"step {step}: model called {call.name}({call.args})")
            tool_started = time.perf_counter()
            result = run_tool(toolbox, call)
            tool_seconds += time.perf_counter() - tool_started
            log(f"         -> {result}")
            results.append((call.name, result))
            call_ids.append(call.id or call.name)
        history.append(ToolResults(results, call_ids))

        # One successful tool call needs no extra model call to be put into words.
        if len(results) == 1:
            call, (name, result) = turn.tool_calls[0], results[0]
            if name == "change_employee":
                message = _change_reply(call.args, result, toolbox)
                if message is not None:
                    log_timing()
                    _remember_conversation(conversation_history, question, message)
                    declined = result.get("status") == "declined"
                    return AgentResult(
                        status=toolbox.state.last_approval or "answered", message=message,
                        payslip=None if declined else toolbox.state.last_payslip)
            else:
                message = _format_reply(name, call.args, result)
                if message is not None:
                    log_timing()
                    _remember_conversation(conversation_history, question, message)
                    return AgentResult(status="answered", message=message, payslip=toolbox.state.last_payslip)
    else:
        log_timing()
        return AgentResult(status="error", message=f"Stopped after {MAX_STEPS} steps without a final answer.")

    log_timing()

    reply = llm.structure(history)

    # Code, not the model, decides these two facts: whether a human approved, and the numbers.
    status = toolbox.state.last_approval or reply.status
    _remember_conversation(conversation_history, question, reply.message)

    return AgentResult(status=status, message=reply.message, payslip=toolbox.state.last_payslip)
