"""Tests for the Penny web server: approval is pause-by-exception and approve-by-replay.

They use the scripted fake model, so they need no API key or internet.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from payroll_agent.tools import DEFAULT_EMPLOYEES
from tests.fake_llm import FakeLLM, call, final
from web.server import create_app

SID = "test-session-1"


def make_client(turns):
    """A client whose sessions each get a FakeLLM playing `turns`. Returns (client, fake models created)."""
    created = []

    def factory(toolbox, log):
        llm = FakeLLM(list(turns))
        created.append(llm)
        return llm

    return TestClient(create_app(llm_factory=factory)), created


def session_dir(client, sid=SID) -> Path:
    return client.app.state.sessions[sid].dir


def chat(client, message="hello", sid=SID):
    return client.post("/api/chat", json={"session_id": sid, "message": message})


def decide(client, approval_id, approve, sid=SID):
    return client.post("/api/approve", json={"session_id": sid, "approval_id": approval_id, "approve": approve})


CHANGE_AGE = [call("change_employee", action="update", name="Thandi", age=66)]


def test_read_only_question_is_answered_with_engine_figures():
    client, _ = make_client([call("employee_pay", name="Thandi", fields=["net"])])
    reply = chat(client, "thandis net pay").json()

    assert reply["status"] == "answered"
    assert reply["payslip"] is not None
    assert f"{reply['payslip']['net_pay']:,.2f}" in reply["message"]  # the figure comes from the engine
    assert any("employee_pay" in line for line in reply["trace"])


@pytest.mark.parametrize("text", ["Hi Penny", "hello", "Hey penny!", "what can you do?", "Penny, who are you"])
def test_greetings_are_answered_by_code_without_the_model(text):
    client, created = make_client([final("should not be used")])
    reply = chat(client, text).json()

    assert reply["status"] == "answered"
    assert reply["message"].startswith("Hi, I'm Penny")
    assert created[0].step_calls == 0


def test_a_question_that_starts_with_hi_still_goes_to_the_model():
    client, created = make_client([call("employee_pay", name="Thandi", fields=["net"])])
    reply = chat(client, "hi, what is thandi's net pay?").json()

    assert created[0].step_calls == 1
    assert reply["payslip"] is not None


def test_a_two_part_question_is_not_cut_short_after_the_first_answer():
    # "tax now AND IF she got 5k more": the reply must wait for the second tool call
    client, created = make_client([
        call("employee_pay", name="Thandi", fields=["paye"]),
        call("employee_pay", name="Thandi", salary=30000, fields=["paye"]),
        final("done"),
    ])
    chat(client, "whats thandis tax and if she got paid 5k more what would it be then")

    assert created[0].step_calls == 3  # first answer, the what-if, then the final reply


def test_team_list_comes_with_a_table_built_by_code():
    client, _ = make_client([call("team_pay", answer="list", figure="paye"), final("Here you go.")])
    reply = chat(client, "tax numbers for everyone").json()

    assert reply["status"] == "answered"
    assert len(reply["table"]) == 5
    assert {"name", "age", "gross", "paye", "uif", "net"} <= set(reply["table"][0])


def test_change_pauses_for_approval_and_saves_nothing():
    client, _ = make_client(CHANGE_AGE)
    reply = chat(client, "change thandi's age to 66").json()
    saved = json.loads((session_dir(client) / "employees.json").read_text())

    assert reply["status"] == "needs_approval"
    assert "30 -> 66" in reply["approval"]["summary"]
    assert next(p for p in saved if p["name"] == "Thandi Nkosi")["age"] == 30  # nothing saved yet
    assert not (session_dir(client) / "audit_log.json").exists()


def test_approve_applies_the_change_without_calling_the_model():
    client, created = make_client(CHANGE_AGE)
    card = chat(client, "change thandi's age to 66").json()["approval"]
    model_calls_before = created[0].step_calls

    reply = decide(client, card["id"], True).json()
    saved = json.loads((session_dir(client) / "employees.json").read_text())
    audit = json.loads((session_dir(client) / "audit_log.json").read_text())

    assert reply["status"] == "approved"
    assert "age from 30 to 66" in reply["message"]
    assert next(p for p in saved if p["name"] == "Thandi Nkosi")["age"] == 66
    assert audit[-1]["action"] == "update" and audit[-1]["new_value"] == 66
    assert created[0].step_calls == model_calls_before == 1  # approving never calls the model


def test_reject_leaves_data_and_audit_untouched():
    client, _ = make_client(CHANGE_AGE)
    card = chat(client, "change thandi's age to 66").json()["approval"]

    reply = decide(client, card["id"], False).json()
    saved = json.loads((session_dir(client) / "employees.json").read_text())

    assert reply["status"] == "approval_declined"
    assert "Nothing was changed" in reply["message"]
    assert next(p for p in saved if p["name"] == "Thandi Nkosi")["age"] == 30
    assert not (session_dir(client) / "audit_log.json").exists()


def test_data_changed_since_the_card_means_nothing_is_applied():
    client, _ = make_client(CHANGE_AGE)
    card = chat(client, "change thandi's age to 66").json()["approval"]
    path = session_dir(client) / "employees.json"
    people = json.loads(path.read_text())
    people[0]["gross_monthly"] = 99000  # someone else edited her salary after the card was shown
    path.write_text(json.dumps(people))

    reply = decide(client, card["id"], True).json()

    assert reply["status"] == "error"
    assert "Nothing was changed" in reply["message"]
    assert json.loads(path.read_text())[0]["age"] == 30  # the age was not changed


def test_an_approval_can_only_be_used_once():
    client, _ = make_client(CHANGE_AGE)
    card = chat(client, "change thandi's age to 66").json()["approval"]
    assert decide(client, card["id"], True).status_code == 200

    assert decide(client, card["id"], True).status_code == 404


def test_a_new_message_cancels_the_open_approval_card():
    client, _ = make_client([*CHANGE_AGE, call("employee_pay", name="Sipho Dlamini", fields=["net"])])
    card = chat(client, "change thandi's age to 66").json()["approval"]
    chat(client, "actually, what is sipho dlamini's net pay?")

    assert decide(client, card["id"], True).status_code == 404


def test_wrong_approval_id_and_unknown_session_are_rejected():
    client, _ = make_client(CHANGE_AGE)
    chat(client, "change thandi's age to 66")

    assert decide(client, "not-the-id", True).status_code == 404
    assert decide(client, "x", True, sid="never-seen-session").status_code == 404
    assert chat(client, "hi", sid="short").status_code == 422  # session id must look like a real id


def test_model_service_error_is_reported_and_changes_nothing():
    class Boom(FakeLLM):
        def step(self, history, tool_names=None):
            err = RuntimeError("rate limited")
            err.status_code = 429
            raise err

    client = TestClient(create_app(llm_factory=lambda toolbox, log: Boom([final()])))
    reply = chat(client, "net pay for thandi").json()

    assert reply["status"] == "error"
    assert "HTTP 429" in reply["message"] and "Nothing was changed" in reply["message"]


def test_sessions_use_their_own_copy_of_the_data():
    original = Path(DEFAULT_EMPLOYEES).read_bytes()
    client, _ = make_client(CHANGE_AGE)
    card = chat(client, "change thandi's age to 66").json()["approval"]
    decide(client, card["id"], True)

    assert Path(DEFAULT_EMPLOYEES).read_bytes() == original  # the repo's data file is never edited


def test_missing_api_key_gives_a_clear_503(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    client = TestClient(create_app())  # the real model factory, with no key

    assert chat(client, "hi").status_code == 503


def test_pages_and_security_headers():
    client, _ = make_client([final()])
    page = client.get("/")

    assert page.status_code == 200
    html = page.text
    assert "<style" not in html and "style=" not in html  # the CSP forbids inline styles
    assert "<script>" not in html  # and inline scripts: all code lives in /static/app.js
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("raw, tidy", [
    ("PAYE is R3 381.00.", "PAYE is R3,381.00."),
    ("PAYE is R3\u202f381.00 a month", "PAYE is R3,381.00 a month"),
    ("R30 000/month and R5 000 more", "R30,000/month and R5,000 more"),
    ("Gross R1\u00a0250\u00a0000.50", "Gross R1,250,000.50"),
    ("already R21,441.88 and R0.00", "already R21,441.88 and R0.00"),
    ("Room 12 and R177.12", "Room 12 and R177.12"),
])
def test_money_in_the_model_words_is_written_one_way(raw, tidy):
    from payroll_agent.agent import tidy_money
    assert tidy_money(raw) == tidy
