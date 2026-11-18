"""Tests for the tools. The expected numbers are worked out by hand from the SARS 2026/27 table,
not copied from the code's own output, so they check the engine independently."""

import json

import pytest

from payroll_agent.tools import build_toolbox

EMPLOYEES = [
    {"name": "Thandi Nkosi", "age": 30, "gross_monthly": 25000, "sdl_exempt": False},
    {"name": "Sipho Dlamini", "age": 45, "gross_monthly": 42000, "sdl_exempt": False},
    {"name": "Sipho Khumalo", "age": 28, "gross_monthly": 18500, "sdl_exempt": False},
]


@pytest.fixture
def toolbox(tmp_path):
    employees = tmp_path / "employees.json"
    employees.write_text(json.dumps(EMPLOYEES))
    return build_toolbox(ask_human=lambda text: True, employees_path=employees, approved_path=tmp_path / "approved.json")


def test_known_payslip_r25000_age_30(toolbox):
    # Annual 300,000 -> 44,118 + 26% of (300,000 - 245,100) = 58,392
    # minus primary rebate 17,820 = 40,572 a year -> 3,381.00 a month
    # UIF is 1% of the R17,712 ceiling = 177.12
    # Net = 25,000 - 3,381.00 - 177.12 = 21,441.88
    result = toolbox.functions["calculate_payslip"](25000, 30, False)
    assert result["paye"] == 3381.00
    assert result["uif_employee"] == 177.12
    assert result["net_pay"] == 21441.88


def test_below_tax_threshold_pays_no_paye(toolbox):
    # R8,000 x 12 = 96,000, which is under the R99,000 threshold for under-65s
    result = toolbox.functions["calculate_payslip"](8000, 23, False)
    assert result["paye"] == 0.0
    assert result["uif_employee"] == 80.0
    assert result["net_pay"] == 7920.0


def test_age_float_from_model_is_accepted(toolbox):
    # Models may send whole numbers as floats (30.0), so the tool must cope
    result = toolbox.functions["calculate_payslip"](25000.0, 30.0, False)
    assert result["paye"] == 3381.00


def test_invalid_inputs_return_error_not_crash(toolbox):
    assert "error" in toolbox.functions["calculate_payslip"](-100, 30, False)
    assert "error" in toolbox.functions["calculate_payslip"](25000, 5, False)


def test_lookup_exact_and_partial_name(toolbox):
    assert toolbox.functions["lookup_employee"]("thandi nkosi")["age"] == 30
    assert toolbox.functions["lookup_employee"]("Thandi")["gross_monthly"] == 25000


def test_lookup_returns_current_payslip(toolbox):
    result = toolbox.functions["lookup_employee"]("Thandi")
    assert {"paye", "uif_employee", "net_pay"} <= result.keys()
    assert result["net_pay"] == round(result["gross_monthly"] - result["paye"] - result["uif_employee"], 2)


def test_lookup_unknown_name_is_an_error(toolbox):
    assert "error" in toolbox.functions["lookup_employee"]("Nobody Here")


def test_lookup_ambiguous_name_lists_candidates(toolbox):
    result = toolbox.functions["lookup_employee"]("Sipho")
    assert "error" in result
    assert "Sipho Dlamini" in result["error"] and "Sipho Khumalo" in result["error"]


def test_change_employee_rejects_unclear_requests(toolbox):
    change = toolbox.model_tools["change_employee"]
    assert "error" in change(action="update", name="Thandi Nkosi")
    assert "error" in change(action="update", name="Thandi Nkosi", age=40, gross_monthly=30000)
    assert "error" in change(action="add", name="New Person", age=30)
    assert "error" in change(action="rename", name="Thandi Nkosi")
    assert "error" in change(action="remove", name="Thandi")  # needs the exact full name
    assert "error" in change(action="explode", name="Thandi Nkosi")


def test_change_action_is_sent_to_the_model_as_an_enum(toolbox):
    from payroll_agent.llm import _function_schema
    schema = _function_schema(toolbox.model_tools["change_employee"])
    action = schema["function"]["parameters"]["properties"]["action"]
    assert action["enum"] == ["approve", "update", "add", "rename", "remove"]
    assert schema["function"]["parameters"]["required"] == ["action", "name"]


def test_exact_rename_or_remove_does_not_strip_a_bare_s(toolbox):
    assert "error" in toolbox.model_tools["change_employee"](action="remove", name="Thandi Nkosis")


def test_model_sees_only_the_three_tools(toolbox):
    assert list(toolbox.model_tools) == ["employee_pay", "team_pay", "change_employee"]


def test_employee_pay_by_name_uses_saved_salary_and_age(toolbox):
    result = toolbox.model_tools["employee_pay"](name="Thandi")
    assert (result["name"], result["age"], result["gross"], result["paye"], result["uif"],
            result["net"], result["sdl"]) == ("Thandi Nkosi", 30, 25000, 3381.0, 177.12, 21441.88, 250.0)
    assert result["period"] == "month" and result["fields"] == []


def test_employee_pay_what_if_overrides_only_the_given_value(toolbox):
    result = toolbox.model_tools["employee_pay"](name="Thandi", age=66)
    assert result["gross"] == 25000 and result["age"] == 66
    assert result["net"] == 22255.63
    assert result["what_if"] is True


def test_employee_pay_without_name_or_salary_is_an_error(toolbox):
    assert "error" in toolbox.model_tools["employee_pay"](age=30)
    assert "error" in toolbox.model_tools["employee_pay"](name="Nobody Here")
    assert "More than one employee" in toolbox.model_tools["employee_pay"](name="Sipho")["error"]


def test_employee_pay_explain_adds_the_breakdown(toolbox):
    assert "annual_paye" in toolbox.model_tools["employee_pay"](name="Thandi", explain=True)


def test_employee_pay_rejects_unknown_figures(toolbox):
    assert "error" in toolbox.model_tools["employee_pay"](name="Thandi", fields=["wage"])


def test_employee_pay_periods_use_the_annual_tax_table(toolbox):
    pay = toolbox.model_tools["employee_pay"]
    assert pay(salary=50000, age=27)["net"] == 38747.3
    assert pay(salary=50000, age=27, period="year")["paye"] == 132907.0
    assert pay(salary=600000, age=27, salary_period="year", period="month")["paye"] == 11075.58
    assert pay(salary=10000, age=27, salary_period="week")["paye"] == 2011.87
    assert pay(salary=1000, age=27, salary_period="day", days_per_week=5, period="month")["paye"] == 2514.33


def test_a_daily_rate_needs_paid_days_per_week(toolbox):
    pay = toolbox.model_tools["employee_pay"]
    assert "days_per_week" in pay(salary=1000, age=27, salary_period="day")["error"]
    assert "days_per_week" in pay(salary=1000, age=27, salary_period="day", days_per_week=9)["error"]
    assert "error" in pay(salary=5000, age=27, period="day")


def test_possessive_without_apostrophe_finds_the_employee(toolbox):
    pay = toolbox.model_tools["employee_pay"]
    assert pay(name="thandis")["net"] == 21441.88
    assert pay(name="Thandi's")["name"] == "Thandi Nkosi"
    assert "More than one employee" in pay(name="siphos")["error"]


def test_a_typo_suggests_the_right_names(toolbox):
    pay = toolbox.model_tools["employee_pay"]
    assert "Did you mean: Thandi Nkosi?" in pay(name="Thandy")["error"]
    error = pay(name="Sipo")["error"]
    assert "Sipho Dlamini" in error and "Sipho Khumalo" in error


def test_team_pay_list_has_everyone_with_their_figures(toolbox):
    result = toolbox.model_tools["team_pay"]()
    people = {p["name"]: p for p in result["employees"]}
    assert result["count"] == 3
    assert people["Sipho Dlamini"]["paye"] == 8304.75
    assert people["Thandi Nkosi"]["net"] == 21441.88 and people["Thandi Nkosi"]["sdl"] == 250.0


def test_team_pay_highest_lowest_count_and_totals(toolbox):
    team = toolbox.model_tools["team_pay"]
    assert team("count") == {"answer": "count", "count": 3}
    assert team("highest", "paye")["names"] == ["Sipho Dlamini"]
    assert team("lowest", "net")["names"] == ["Sipho Khumalo"]
    assert team("highest")["figure"] == "gross"  # "all" falls back to gross when ranking
    totals = team("total")["totals"]
    assert totals["paye"] == pytest.approx(3381.0 + 8304.75 + 1845.0)
    assert totals["sdl"] == pytest.approx(250.0 + 420.0 + 185.0)
    assert "error" in team("total", "age")
    assert "error" in team("sideways")


def test_figures_and_periods_are_sent_to_the_model_as_enums(toolbox):
    from payroll_agent.llm import _function_schema
    properties = _function_schema(toolbox.model_tools["employee_pay"])["function"]["parameters"]["properties"]
    assert properties["fields"] == {"type": "array", "items": {
        "type": "string", "enum": ["gross", "paye", "uif", "sdl", "net", "age"]}}
    assert properties["period"]["enum"] == ["month", "year", "week", "day"]
