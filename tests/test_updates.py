"""Tests for the update_employee tool: changes need human approval, are validated, and are logged."""

import json

from payroll_agent.agent import run_agent
from payroll_agent.tools import build_toolbox
from tests.fake_llm import FakeLLM, call, final

EMPLOYEES = [
    {"name": "Thandi Nkosi", "age": 30, "gross_monthly": 25000, "sdl_exempt": False},
    {"name": "Sipho Dlamini", "age": 45, "gross_monthly": 42000, "sdl_exempt": False},
    {"name": "Sipho Khumalo", "age": 28, "gross_monthly": 18500, "sdl_exempt": False},
]


def setup(tmp_path, approve=True, prompts=None):
    employees = tmp_path / "employees.json"
    employees.write_text(json.dumps(EMPLOYEES))
    audit = tmp_path / "audit.json"

    def ask_human(text):
        if prompts is not None:
            prompts.append(text)
        return approve

    toolbox = build_toolbox(
        ask_human=ask_human,
        employees_path=employees,
        approved_path=tmp_path / "approved.json",
        audit_path=audit,
    )
    return toolbox, employees, audit


def test_declined_change_leaves_data_and_audit_untouched(tmp_path):
    toolbox, employees, audit = setup(tmp_path, approve=False)
    result = toolbox.functions["update_employee"]("Thandi", "age", 66)

    assert result["status"] == "declined"
    assert json.loads(employees.read_text()) == EMPLOYEES
    assert not audit.exists()


def test_approved_change_is_saved_and_logged(tmp_path):
    toolbox, employees, audit = setup(tmp_path, approve=True)
    result = toolbox.functions["update_employee"]("Thandi", "age", 66)

    assert result["status"] == "approved"
    saved = {e["name"]: e for e in json.loads(employees.read_text())}
    assert saved["Thandi Nkosi"]["age"] == 66
    assert saved["Sipho Dlamini"]["age"] == 45  # nobody else changed

    log = json.loads(audit.read_text())
    assert len(log) == 1
    assert log[0]["old_value"] == 30 and log[0]["new_value"] == 66 and log[0]["field"] == "age"


def test_age_66_adds_the_secondary_rebate(tmp_path):
    # Hand-worked: tax before rebates 58,392, rebates 17,820 + 9,765 = 27,585
    # -> 30,807 a year = 2,567.25 a month. Net = 25,000 - 2,567.25 - 177.12 = 22,255.63
    toolbox, _, _ = setup(tmp_path, approve=True)
    toolbox.functions["update_employee"]("Thandi", "age", 66)

    assert toolbox.state.last_payslip.paye == 2567.25
    assert toolbox.state.last_payslip.net_pay == 22255.63


def test_approval_prompt_shows_before_and_after(tmp_path):
    prompts = []
    toolbox, _, _ = setup(tmp_path, approve=False, prompts=prompts)
    toolbox.functions["update_employee"]("Thandi", "age", 66)

    assert "30" in prompts[0] and "66" in prompts[0]
    assert "21,441.88" in prompts[0] and "22,255.63" in prompts[0]


def test_only_allowed_fields_can_be_changed(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    for bad_field in ("name", "net_pay", "sdl_exempt"):
        assert "error" in toolbox.functions["update_employee"]("Thandi", bad_field, 1)
    assert json.loads(employees.read_text()) == EMPLOYEES


def test_invalid_values_are_rejected(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    update = toolbox.functions["update_employee"]
    assert "error" in update("Thandi", "age", 5)          # engine says age must be 16 to 100
    assert "error" in update("Thandi", "age", 66.5)       # not a whole number
    assert "error" in update("Thandi", "gross_monthly", -100)
    assert "error" in update("Thandi", "gross_monthly", 0)
    assert json.loads(employees.read_text()) == EMPLOYEES


def test_ambiguous_or_unknown_name_changes_nothing(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    assert "error" in toolbox.functions["update_employee"]("Sipho", "gross_monthly", 25000)
    assert "error" in toolbox.functions["update_employee"]("Nobody", "age", 40)
    assert json.loads(employees.read_text()) == EMPLOYEES


def test_agent_applies_change_and_returns_new_numbers(tmp_path):
    toolbox, employees, _ = setup(tmp_path, approve=True)
    llm = FakeLLM([call("change_employee", action="update", name="Thandi Nkosi", age=66), final()])
    result = run_agent("Change Thandi's age to 66", llm, toolbox)

    assert result.status == "approved"
    assert result.payslip.net_pay == 22255.63


def test_what_if_does_not_change_data(tmp_path):
    toolbox, employees, audit = setup(tmp_path, approve=True)
    llm = FakeLLM([call("employee_pay", salary=25000, age=66), final()])
    result = run_agent("What would Thandi's pay be at 66?", llm, toolbox)

    assert result.payslip.net_pay == 22255.63
    assert json.loads(employees.read_text()) == EMPLOYEES
    assert not audit.exists()


def test_agent_declined_change_reports_declined(tmp_path):
    toolbox, employees, _ = setup(tmp_path, approve=False)
    llm = FakeLLM([call("change_employee", action="update", name="Thandi Nkosi", age=66), final()])
    result = run_agent("Change Thandi's age to 66", llm, toolbox)

    assert result.status == "approval_declined"
    assert json.loads(employees.read_text()) == EMPLOYEES
