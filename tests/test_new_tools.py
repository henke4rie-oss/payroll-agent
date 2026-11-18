"""Tests for the edge cases: typos, renames, adds, removes, and whole-company questions."""

import json

import pytest

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

    toolbox = build_toolbox(ask_human=ask_human, employees_path=employees,
                            approved_path=tmp_path / "approved.json", audit_path=audit)
    return toolbox, employees, audit


def saved(employees):
    return json.loads(employees.read_text())


# ---------- typos ----------

def test_typo_in_name_suggests_the_close_match(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    result = toolbox.functions["lookup_employee"]("Thandy Nkosi")
    assert "error" in result and "Thandi Nkosi" in result["error"]


def test_typo_never_changes_anything(tmp_path):
    toolbox, employees, audit = setup(tmp_path)
    assert "error" in toolbox.functions["update_employee"]("Thandy Nkosi", "age", 66)
    assert saved(employees) == EMPLOYEES and not audit.exists()


# ---------- rename ----------

def test_rename_fixes_a_typo_and_logs_it(tmp_path):
    toolbox, employees, audit = setup(tmp_path)
    result = toolbox.functions["rename_employee"]("Thandi Nkosi", "Thandi Nkosie")

    assert result["status"] == "approved"
    assert "Thandi Nkosie" in [p["name"] for p in saved(employees)]
    log = json.loads(audit.read_text())
    assert log[0]["old_name"] == "Thandi Nkosi" and log[0]["new_name"] == "Thandi Nkosie"


def test_rename_needs_the_full_name(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    assert "error" in toolbox.functions["rename_employee"]("Thandi", "Thandi Nkosie")
    assert saved(employees) == EMPLOYEES


def test_rename_cannot_collide_with_another_employee(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    result = toolbox.functions["rename_employee"]("Sipho Khumalo", "sipho dlamini")  # different case
    assert "error" in result
    assert saved(employees) == EMPLOYEES


def test_rename_rejects_bad_names(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    for bad in ("", "   ", "Thandi123", "<script>", "A" * 61, "Thandi Nkosi"):
        assert "error" in toolbox.functions["rename_employee"]("Thandi Nkosi", bad), bad
    assert saved(employees) == EMPLOYEES


def test_rename_declined_changes_nothing(tmp_path):
    toolbox, employees, audit = setup(tmp_path, approve=False)
    assert toolbox.functions["rename_employee"]("Thandi Nkosi", "Thandi Nkosie")["status"] == "declined"
    assert saved(employees) == EMPLOYEES and not audit.exists()


def test_rename_allows_apostrophes_hyphens_and_accents(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    toolbox.functions["rename_employee"]("Thandi Nkosi", "  Thandi   O'Brien-Nkosi  ")
    assert saved(employees)[0]["name"] == "Thandi O'Brien-Nkosi"  # extra spaces tidied
    toolbox.functions["rename_employee"]("Sipho Dlamini", "Siphó Dlamini")
    assert "Siphó Dlamini" in [p["name"] for p in saved(employees)]


# ---------- add ----------

def test_add_employee_saves_and_logs(tmp_path):
    toolbox, employees, audit = setup(tmp_path)
    result = toolbox.functions["add_employee"]("Zanele Mkhize", 35, 30000)

    assert result["status"] == "approved"
    assert saved(employees)[-1] == {"name": "Zanele Mkhize", "age": 35, "gross_monthly": 30000.0, "sdl_exempt": False}
    assert json.loads(audit.read_text())[0]["action"] == "add"


def test_add_rejects_duplicates_bad_names_and_bad_values(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    add = toolbox.functions["add_employee"]
    assert "error" in add("thandi nkosi", 30, 20000)   # duplicate, different case
    assert "error" in add("", 30, 20000)
    assert "error" in add("Zanele", 12, 20000)         # engine rejects age under 16
    assert "error" in add("Zanele", 30.5, 20000)       # age must be whole
    assert "error" in add("Zanele", 30, 0)
    assert "error" in add("Zanele", 30, -5)
    assert saved(employees) == EMPLOYEES


def test_add_declined_changes_nothing(tmp_path):
    toolbox, employees, _ = setup(tmp_path, approve=False)
    assert toolbox.functions["add_employee"]("Zanele Mkhize", 35, 30000)["status"] == "declined"
    assert saved(employees) == EMPLOYEES


def test_added_employee_can_then_be_found(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    toolbox.functions["add_employee"]("Zanele Mkhize", 35, 30000)
    assert toolbox.functions["lookup_employee"]("Zanele")["gross_monthly"] == 30000.0


# ---------- remove ----------

def test_remove_needs_full_name_and_keeps_a_copy_in_the_log(tmp_path):
    toolbox, employees, audit = setup(tmp_path)
    assert "error" in toolbox.functions["remove_employee"]("Thandi")  # partial name refused
    assert saved(employees) == EMPLOYEES

    result = toolbox.functions["remove_employee"]("thandi nkosi")  # full name, any case
    assert result["status"] == "approved"
    assert "Thandi Nkosi" not in [p["name"] for p in saved(employees)]
    assert json.loads(audit.read_text())[0]["removed_record"]["gross_monthly"] == 25000


def test_remove_declined_keeps_the_employee(tmp_path):
    toolbox, employees, _ = setup(tmp_path, approve=False)
    assert toolbox.functions["remove_employee"]("Thandi Nkosi")["status"] == "declined"
    assert saved(employees) == EMPLOYEES


def test_remove_unknown_name_removes_nobody(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    assert "error" in toolbox.functions["remove_employee"]("Nobody Here")
    assert saved(employees) == EMPLOYEES


# ---------- whole-company questions ----------

def test_list_employees(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    result = toolbox.functions["list_employees"]()
    assert result["count"] == 3
    assert {e["name"] for e in result["employees"]} == {"Thandi Nkosi", "Sipho Dlamini", "Sipho Khumalo"}


def test_payroll_summary_equals_the_sum_of_individual_payslips(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    summary = toolbox.functions["payroll_summary"]()
    slips = [toolbox.functions["calculate_payslip"](p["gross_monthly"], p["age"], False) for p in EMPLOYEES]

    assert summary["employee_count"] == 3
    assert summary["total_gross"] == 85500.0
    assert summary["total_net_pay"] == pytest.approx(sum(s["net_pay"] for s in slips))
    assert summary["total_paye"] == pytest.approx(sum(s["paye"] for s in slips))


def test_payroll_summary_with_no_employees(tmp_path):
    toolbox, employees, _ = setup(tmp_path)
    employees.write_text("[]")
    assert toolbox.functions["payroll_summary"]()["total_net_pay"] == 0.0


def test_explain_payslip_shows_the_age_rebate_effect(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    young = toolbox.functions["explain_payslip"](25000, 30, False)
    older = toolbox.functions["explain_payslip"](25000, 66, False)

    assert young["total_rebates"] == 17820.0 and young["secondary_rebate"] == 0.0
    assert older["total_rebates"] == 27585.0 and older["secondary_rebate"] == 9765.0
    assert older["monthly_paye"] == 2567.25
    assert "error" in toolbox.functions["explain_payslip"](25000, 5, False)


# ---------- through the agent ----------

def test_agent_typo_flow_asks_instead_of_guessing(tmp_path):
    from payroll_agent.schemas import ModelReply
    toolbox, employees, _ = setup(tmp_path)
    llm = FakeLLM(
        [call("change_employee", action="update", name="Thandy Nkosi", age=66), final()],
        reply=ModelReply(status="needs_clarification", message="Did you mean Thandi Nkosi?"),
    )
    result = run_agent("Change Thandy's age to 66", llm, toolbox)

    assert result.status == "needs_clarification"
    assert saved(employees) == EMPLOYEES


def test_agent_rename_then_lookup_uses_the_new_name(tmp_path):
    toolbox, _, _ = setup(tmp_path)
    llm = FakeLLM([
        call("change_employee", action="rename", name="Thandi Nkosi", new_name="Thandi Nkosie"),
        call("employee_pay", name="Thandi Nkosie"),
        final(),
    ])
    result = run_agent("Fix the spelling of Thandi Nkosi to Thandi Nkosie", llm, toolbox)
    assert result.status == "approved"
