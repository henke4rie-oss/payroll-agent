"""Run the payroll agent in the terminal:  python main.py   (add --stats to see model calls, tokens and timings)"""

from datetime import datetime
import os
import sys
import time

from dotenv import load_dotenv

from payroll_agent.agent import SYSTEM_PROMPT, run_agent
from payroll_agent.llm import DEFAULT_MODEL, ChatLLM
from payroll_agent.tools import build_toolbox

STATS = "--stats" in sys.argv or os.getenv("PAYROLL_STATS") == "1"


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def log(message: str) -> None:
    """Debug detail: only shown with --stats."""
    if STATS:
        print(f"[{timestamp()}] {message}")


def say(message: str) -> None:
    print(message)


def ask_human(summary: str) -> bool:
    """The human-in-the-loop step: nothing is approved unless a person types y."""
    say(f"\nPlease confirm: {summary}")
    return input("Approve? [y/N] ").strip().lower() == "y"


def main() -> None:
    load_dotenv()  # reads LLM_API_KEY (and optional LLM_MODEL, LLM_BASE_URL) from your .env file
    if not os.getenv("LLM_API_KEY"):
        sys.exit("Missing LLM_API_KEY. Copy .env.example to .env and add your key.")

    toolbox = build_toolbox(ask_human=ask_human)
    llm = ChatLLM(
        tools=list(toolbox.model_tools.values()),
        system_instruction=SYSTEM_PROMPT,
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        log=log,
    )

    say(f"Payroll agent ready ({llm.model}). Type 'quit' to exit.")
    conversation_history = []
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("", "quit", "exit"):
            break
        request_started = time.perf_counter()
        log(f"Request received: {question}")
        try:
            result = run_agent(
                question, llm, toolbox, log=log, conversation_history=conversation_history)
        except Exception as err:  # e.g. rate limit after all retries, or no internet
            status_code = getattr(err, "status_code", None)
            body = getattr(err, "body", None)
            if status_code is not None:
                say(f"The model service returned HTTP {status_code}. Try again in a moment.")
                log(str(body if body is not None else err))
                if status_code == 410:
                    log("Check LLM_MODEL and LLM_BASE_URL; retrying a 410 will not resolve it.")
            else:
                say(f"Something went wrong: {err}")
            elapsed = time.perf_counter() - request_started
            log(f"Request failed after {elapsed:.3f}s. Nothing was changed.")
            say("Nothing was changed.")
            continue
        elapsed = time.perf_counter() - request_started
        log(f"Agent [{result.status}] (request-to-reply {elapsed:.3f}s)")
        say(f"Agent: {result.message}")


if __name__ == "__main__":
    main()
