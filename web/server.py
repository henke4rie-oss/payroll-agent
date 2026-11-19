"""Penny on the web: a small FastAPI server around the payroll agent.

The hard part is human approval. In the terminal the agent simply waits for you to type y. A web
request can't wait, so approval works like this ("pause by exception, approve by replay"):

1. The model asks for a change. The tool checks it and works out the before/after, then calls
   ask_human(summary). Here, ask_human raises ApprovalNeeded, so the agent stops and nothing is saved.
2. The server remembers that exact tool call and sends the summary to the browser as an approval card.
3. The person clicks Approve or Reject (a second request, /api/approve).
4. Approve runs the SAME tool call again in plain code, with no model call, and approves only the exact
   summary the person saw. If the data changed in between, the summary differs and nothing is applied.

The model never sees or decides the approval, and every rule inside the tools (validation, exact names,
audit log) still runs, because it is the same code as in the terminal version.

Each browser session gets its own temporary copy of the employee data, so demo edits never touch the repo.
"""

import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from payroll_agent.agent import SYSTEM_PROMPT, ApprovalNeeded, apply_approved, remember, run_agent
from payroll_agent.llm import DEFAULT_MODEL, ChatLLM, ToolCall
from payroll_agent.schemas import PayslipSummary
from payroll_agent.tools import DEFAULT_EMPLOYEES, Toolbox, build_toolbox

STATIC_DIR = Path(__file__).parent / "static"
SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 50  # bounds memory and temp disk use
SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]{8,64}$"
MAX_TRACE_LINE = 300


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    message: str = Field(min_length=1, max_length=500)


class ApproveRequest(BaseModel):
    session_id: str = Field(pattern=SESSION_ID_PATTERN)
    approval_id: str = Field(min_length=1, max_length=64)
    approve: bool


class ApprovalCard(BaseModel):
    id: str
    summary: str


class Reply(BaseModel):
    status: str  # answered | needs_clarification | needs_approval | approved | approval_declined | error
    message: str
    payslip: Optional[PayslipSummary] = None
    table: Optional[list[dict]] = None  # everyone's figures, straight from the tools (not typed by the model)
    approval: Optional[ApprovalCard] = None
    trace: list[str] = []  # what the agent did, so the page can show "How Penny got this"


class TeamReply(BaseModel):
    employees: list[dict]
    totals: dict


class ActivityReply(BaseModel):
    entries: list[dict]


def _rands(value) -> str:
    return f"R{float(value):,.2f}"


def describe_entry(entry: dict) -> dict:
    """One audit-log (or approved-payslip) record in plain words for the Activity tab."""
    action = entry.get("action", "approve")
    when = entry.get("changed_at") or entry.get("approved_at") or ""
    employee = entry.get("employee", "")
    if action == "update":
        salary = entry.get("field") == "gross_monthly"
        show = _rands if salary else str
        label = "Salary" if salary else str(entry.get("field", "Value")).capitalize()
        text = f"{label} {show(entry.get('old_value'))} \u2192 {show(entry.get('new_value'))}"
    elif action == "add":
        text = f"Added (age {entry.get('age')}, {_rands(entry.get('gross_monthly', 0))} a month)"
    elif action == "rename":
        employee = entry.get("new_name", "")
        text = f"Renamed from {entry.get('old_name', '')}"
    elif action == "remove":
        text = "Removed (the details are kept in the audit log)"
    else:
        text = "Payslip approved"
    return {"when": when, "employee": employee, "text": text}


def read_activity(session_dir: Path, limit: int = 50) -> list[dict]:
    """Everything a person approved in this session, newest first."""
    entries: list[dict] = []
    for name in ("audit_log.json", "approved_payslips.json"):
        path = session_dir / name
        if path.exists():
            try:
                entries.extend(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    described = [describe_entry(e) for e in entries]
    described.sort(key=lambda e: e["when"], reverse=True)
    return described[:limit]


@dataclass
class Pending:
    id: str
    call: ToolCall
    summary: str
    question: str


LlmFactory = Callable[[Toolbox, Callable[[str], None]], object]


def default_llm_factory(toolbox: Toolbox, log: Callable[[str], None]) -> ChatLLM:
    return ChatLLM(
        tools=list(toolbox.model_tools.values()),
        system_instruction=SYSTEM_PROMPT,
        model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
        log=log,
    )


class Session:
    """One browser's conversation: its own data copy, toolbox, model client, history and open approval."""

    def __init__(self, llm_factory: LlmFactory):
        self.dir = Path(tempfile.mkdtemp(prefix="penny-"))
        try:
            shutil.copy(DEFAULT_EMPLOYEES, self.dir / "employees.json")
            self.history: list = []
            self.pending: Optional[Pending] = None
            self.replay_summary: Optional[str] = None  # set only while an approved call is being replayed
            self.trace: list[str] = []
            self.lock = threading.Lock()
            self.last_used = time.time()
            self.toolbox = build_toolbox(
                ask_human=self._ask_human,
                employees_path=self.dir / "employees.json",
                approved_path=self.dir / "approved_payslips.json",
                audit_path=self.dir / "audit_log.json",
            )
            self.llm = llm_factory(self.toolbox, self.log)
        except Exception:
            self.close()
            raise

    def log(self, message: str) -> None:
        self.trace.append(message if len(message) <= MAX_TRACE_LINE else message[:MAX_TRACE_LINE] + "...")

    def _ask_human(self, summary: str) -> bool:
        if self.replay_summary is None:
            raise ApprovalNeeded(summary)  # pause: a person has to click first
        return summary == self.replay_summary  # replay: approve only what the person actually saw

    def close(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _friendly_error(err: Exception) -> str:
    status_code = getattr(err, "status_code", None)
    if status_code is not None:
        return f"The model service returned HTTP {status_code}. Try again in a moment. Nothing was changed."
    return "Something went wrong on my side. Nothing was changed."


def create_app(llm_factory: LlmFactory = default_llm_factory) -> FastAPI:
    app = FastAPI(title="Penny", docs_url=None, redoc_url=None, openapi_url=None)
    sessions: dict[str, Session] = {}
    registry_lock = threading.Lock()
    app.state.sessions = sessions

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "frame-ancestors 'none'; base-uri 'none'")
        return response

    def get_session(session_id: str, create: bool) -> Session:
        now = time.time()
        with registry_lock:
            for key in [k for k, s in sessions.items() if now - s.last_used > SESSION_TTL_SECONDS]:
                sessions.pop(key).close()
            session = sessions.get(session_id)
            if session is None:
                if not create:
                    raise HTTPException(404, "Unknown session. Start a new chat.")
                if len(sessions) >= MAX_SESSIONS:
                    oldest = min(sessions, key=lambda k: sessions[k].last_used)
                    sessions.pop(oldest).close()
                try:
                    session = Session(llm_factory)
                except ValueError:  # ChatLLM raises this when LLM_API_KEY is missing
                    raise HTTPException(503, "Penny isn't set up yet: the server has no LLM_API_KEY.")
                sessions[session_id] = session
            session.last_used = now
            return session

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/chat", response_model=Reply)
    def chat(req: ChatRequest) -> Reply:
        session = get_session(req.session_id, create=True)
        with session.lock:
            session.pending = None  # a new message cancels any approval card that was still open
            session.replay_summary = None
            session.trace = []
            try:
                result = run_agent(req.message, session.llm, session.toolbox, log=session.log,
                                   conversation_history=session.history)
            except ApprovalNeeded as need:
                pending = Pending(secrets.token_urlsafe(8), need.call, need.summary, req.message)
                session.pending = pending
                session.log(f"paused: waiting for a person to approve {need.call.name}({need.call.args})")
                return Reply(status="needs_approval", message=f"Please confirm: {need.summary}",
                             approval=ApprovalCard(id=pending.id, summary=need.summary), trace=session.trace)
            except Exception as err:  # e.g. rate limit after all retries, or no internet
                session.log(f"error: {type(err).__name__}")
                return Reply(status="error", message=_friendly_error(err), trace=session.trace)
            return Reply(status=result.status, message=result.message, payslip=result.payslip,
                         table=result.table, trace=session.trace)

    @app.post("/api/approve", response_model=Reply)
    def approve(req: ApproveRequest) -> Reply:
        session = get_session(req.session_id, create=False)
        with session.lock:
            pending = session.pending
            if pending is None or not secrets.compare_digest(pending.id, req.approval_id):
                raise HTTPException(404, "That approval is no longer pending.")
            session.pending = None
            session.trace = []
            if not req.approve:
                message = "Cancelled. Nothing was changed."
                remember(session.history, pending.question, message)
                session.log("a person rejected the change; nothing was saved")
                return Reply(status="approval_declined", message=message, trace=session.trace)
            session.log(f"a person approved: running {pending.call.name}({pending.call.args}) again in code, no model call")
            session.replay_summary = pending.summary
            try:
                result = apply_approved(session.toolbox, pending.call)
            finally:
                session.replay_summary = None
            remember(session.history, pending.question, result.message)
            session.log(f"result: {result.status}")
            return Reply(status=result.status, message=result.message, payslip=result.payslip, trace=session.trace)

    @app.get("/api/team", response_model=TeamReply)
    def team(session_id: str = Query(pattern=SESSION_ID_PATTERN)) -> TeamReply:
        """Everyone's monthly figures from the tax engine. Read-only: no model call, nothing changes."""
        session = get_session(session_id, create=True)
        with session.lock:
            return TeamReply(employees=session.toolbox.functions["list_employees"]()["employees"],
                             totals=session.toolbox.functions["payroll_summary"]())

    @app.get("/api/activity", response_model=ActivityReply)
    def activity(session_id: str = Query(pattern=SESSION_ID_PATTERN)) -> ActivityReply:
        """The changes a person approved in this session, newest first. Read-only."""
        session = get_session(session_id, create=True)
        with session.lock:
            return ActivityReply(entries=read_activity(session.dir))

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


load_dotenv()  # reads LLM_API_KEY from .env for `uvicorn web.server:app`
app = create_app()
