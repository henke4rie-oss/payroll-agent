"""Run the eval cases against the real model:  python -m evals.run [--only id1,id2] [--save NAME]

Costs real tokens (about 40k for all cases). Every case runs on a copy of the employee data,
and the approval prompt always says no, so the real data is never changed.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from payroll_agent.agent import SYSTEM_PROMPT, run_agent
from payroll_agent.llm import DEFAULT_MODEL, ChatLLM
from payroll_agent.tools import build_toolbox

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "payroll_agent" / "employees.json"
TIMING = re.compile(r"timing: (\d+) model calls .*tokens in=(\d+) out=(\d+)")
MONEY = re.compile(r"R\d")
TOKENS_PER_MINUTE = int(os.getenv("EVAL_TPM", "6000"))  # stays under the free tier limit


def grade(case: dict, reply: str) -> list[str]:
    """Return the failed checks (an empty list means pass)."""
    text = reply.lower()
    failures = []
    for needle in case.get("contains", []):
        if needle.lower() not in text:
            failures.append(f"missing {needle!r}")
    for needle in case.get("not_contains", []):
        if needle.lower() in text:
            failures.append(f"should not contain {needle!r}")
    if case.get("any_of") and not any(n.lower() in text for n in case["any_of"]):
        failures.append(f"none of {case['any_of']!r}")
    if case.get("no_money") and MONEY.search(reply):
        failures.append("invented a money amount")
    return failures


def run_case(case: dict, llm) -> dict:
    workdir = Path(tempfile.mkdtemp())
    try:
        shutil.copy(DATA, workdir / "employees.json")
        toolbox = build_toolbox(
            ask_human=lambda summary: False,
            employees_path=workdir / "employees.json",
            approved_path=workdir / "approved.json",
            audit_path=workdir / "audit.json",
        )
        history: list = []
        lines: list[str] = []
        reply = ""
        for turn in case["turns"]:
            reply = run_agent(turn, llm, toolbox, log=lines.append, conversation_history=history).message
        calls = tokens_in = tokens_out = 0
        for line in lines:
            match = TIMING.search(line)
            if match:
                calls += int(match.group(1))
                tokens_in += int(match.group(2))
                tokens_out += int(match.group(3))
        return {"id": case["id"], "reply": reply, "failures": grade(case, reply),
                "model_calls": calls, "tokens_in": tokens_in, "tokens_out": tokens_out}
    except Exception as err:  # an API error counts as a failure, and the run carries on
        return {"id": case["id"], "reply": "", "failures": [f"error: {err}"],
                "model_calls": 0, "tokens_in": 0, "tokens_out": 0}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def wait_for_budget(recent: list[tuple[float, int]]) -> None:
    """Sleep until the tokens used in the last 60 seconds leave room for the next case."""
    while True:
        now = time.monotonic()
        recent[:] = [(t, n) for t, n in recent if now - t < 60]
        used = sum(n for _, n in recent)
        if used < TOKENS_PER_MINUTE:
            return
        oldest = min(t for t, _ in recent)
        delay = max(1.0, 60 - (now - oldest))
        print(f"waiting {delay:.0f}s (rate limit: {used} of {TOKENS_PER_MINUTE} tokens used this minute)")
        time.sleep(delay)


def main(argv: list[str]) -> None:
    load_dotenv(ROOT.parent / ".env")
    only = None
    save = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
    if "--save" in argv:
        save = argv[argv.index("--save") + 1]

    cases = json.loads((ROOT / "cases.json").read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if c["id"] in only]

    import os
    toolbox = build_toolbox(ask_human=lambda summary: False)
    llm = ChatLLM(
        tools=list(toolbox.model_tools.values()),
        system_instruction=SYSTEM_PROMPT,
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
    )

    results = []
    recent: list[tuple[float, int]] = []  # (finish time, tokens used) for the last minute
    for case in cases:
        wait_for_budget(recent)
        result = run_case(case, llm)
        results.append(result)
        recent.append((time.monotonic(), result["tokens_in"] + result["tokens_out"]))
        status = "PASS" if not result["failures"] else "FAIL"
        print(f"{status} {result['id']:<20} calls={result['model_calls']} "
              f"tokens={result['tokens_in']}/{result['tokens_out']}")
        if result["failures"]:
            print(f"      reply: {result['reply']}")
            print(f"      failed: {'; '.join(result['failures'])}")

    passed = sum(1 for r in results if not r["failures"])
    print(f"\n{passed}/{len(results)} passed | "
          f"tokens in={sum(r['tokens_in'] for r in results)} out={sum(r['tokens_out'] for r in results)} | "
          f"model calls={sum(r['model_calls'] for r in results)}")
    if save:
        out = ROOT / "results" / f"{save}.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
