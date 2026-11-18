"""Tests for the agent loop, using a scripted fake model (no API key needed)."""

import json

import pytest

from payroll_agent.agent import MAX_STEPS, run_agent
from payroll_agent.llm import ModelTurn, ToolResults, UserMessage
from payroll_agent.schemas import ModelReply
from payroll_agent.tools import build_toolbox
from tests.fake_llm import FakeLLM, call, final

EMPLOYEES = [{"name": "Thandi Nkosi", "age": 30, "gross_monthly": 25000, "sdl_exempt": False}]


def make_toolbox(tmp_path, approve: bool):
    employees = tmp_path / "employees.json"
    employees.write_text(json.dumps(EMPLOYEES))
    approved = tmp_path / "approved.json"
    toolbox = build_toolbox(ask_human=lambda text: approve, employees_path=employees, approved_path=approved)
    return toolbox, approved


def test_change_requests_and_their_follow_ups_are_offered_every_tool(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([final("To what age?"), final("ok")])
    conversation_history = []
    run_agent("Please update Thandi's age", llm, toolbox, conversation_history=conversation_history)
    run_agent("66", llm, toolbox, conversation_history=conversation_history)

    assert llm.step_tool_names == [None, None]


def test_approval_declined_writes_nothing_and_code_sets_status(tmp_path):
    toolbox, approved = make_toolbox(tmp_path, approve=False)
    # The model "says" answered, but the human said no, so the code overrides the status
    llm = FakeLLM([call("change_employee", action="approve", name="Thandi Nkosi"), final()],
                  reply=ModelReply(status="answered", message="All done"))
    result = run_agent("Approve Thandi's payslip", llm, toolbox)

    assert result.status == "approval_declined"
    assert not approved.exists()


def test_approval_approved_saves_a_record(tmp_path):
    toolbox, approved = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([call("change_employee", action="approve", name="Thandi Nkosi"), final()])
    result = run_agent("Approve Thandi's payslip", llm, toolbox)

    assert result.status == "approved"
    records = json.loads(approved.read_text())
    assert len(records) == 1 and records[0]["employee"] == "Thandi Nkosi"
    assert records[0]["net_pay"] == 21441.88


def test_approval_tool_ignores_numbers_the_model_invents(tmp_path):
    toolbox, approved = make_toolbox(tmp_path, approve=True)
    # Unknown made-up arguments are rejected, not approved.
    llm = FakeLLM([call("change_employee", action="approve", name="Thandi Nkosi", net_pay=999999), final()])
    result = run_agent("Approve Thandi's payslip", llm, toolbox)

    assert not approved.exists()
    assert result.status != "approved"


def test_approval_recalculates_even_if_the_model_sends_a_salary(tmp_path):
    toolbox, approved = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([call("change_employee", action="approve", name="Thandi Nkosi", gross_monthly=999999), final()])
    run_agent("Approve Thandi's payslip", llm, toolbox)

    assert json.loads(approved.read_text())[0]["gross_monthly"] == 25000


def test_unknown_tool_does_not_crash(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([call("delete_everything"), final()])
    result = run_agent("do something odd", llm, toolbox)

    sent_back = [item for item in llm.history_seen_by_structure if isinstance(item, ToolResults)][0]
    assert "Unknown tool" in sent_back.results[0][1]["error"]
    assert result.status == "answered"


def test_unknown_employee_error_goes_back_to_the_model(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM(
        [call("employee_pay", name="Nobody"), final()],
        reply=ModelReply(status="needs_clarification", message="Who do you mean?"),
    )
    result = run_agent("What does Nobody earn?", llm, toolbox)

    assert result.status == "needs_clarification"
    assert result.payslip is None


def test_step_limit_stops_a_looping_model(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([call("employee_pay", name="Thandi", explain=True)])  # asks for a tool forever
    result = run_agent("loop", llm, toolbox)

    assert result.status == "error"
    assert llm.step_calls == MAX_STEPS


def test_follow_up_questions_keep_recent_conversation_context(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM(
        [final("What age should I use?"), final("Should I use age 27 and R50,000?")],
        reply=ModelReply(status="needs_clarification", message="What age should I use?"),
    )
    conversation_history = []

    run_agent(
        "How much tax would someone earning R50,000 pay?",
        llm,
        toolbox,
        conversation_history=conversation_history,
    )
    run_agent("27", llm, toolbox, conversation_history=conversation_history)

    second_turn_history = llm.step_histories[1]
    assert [item.text for item in second_turn_history if isinstance(item, UserMessage)] == [
        "How much tax would someone earning R50,000 pay?",
        "27",
    ]
    assert any(
        getattr(item, "text", None) == "What age should I use?"
        for item in second_turn_history
    )


def test_token_usage_is_added_up_per_request(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([
        ModelTurn(text="", tool_calls=[call("employee_pay", name="Thandi", explain=True).tool_calls[0]], usage=(100, 5)),
        ModelTurn(text="done", usage=(120, 7)),
    ])
    lines = []
    run_agent("Why is Thandi's tax this amount?", llm, toolbox, log=lines.append)

    assert any("tokens in=220 out=12" in line for line in lines)


def test_conversation_memory_is_capped_at_eight_messages(tmp_path):
    toolbox, _ = make_toolbox(tmp_path, approve=True)
    llm = FakeLLM([final("ok")])
    conversation_history = []
    for _ in range(6):
        run_agent("Who earns the most?", llm, toolbox, conversation_history=conversation_history)

    assert len(conversation_history) == 8


TEAM = [
    {"name": "Thandi Nkosi", "age": 30, "gross_monthly": 25000, "sdl_exempt": False},
    {"name": "Sipho Dlamini", "age": 45, "gross_monthly": 42000, "sdl_exempt": False},
    {"name": "Lerato Mokoena", "age": 64, "gross_monthly": 8000, "sdl_exempt": False},
]


def make_team_toolbox(tmp_path, approve: bool = True):
    employees = tmp_path / "employees.json"
    employees.write_text(json.dumps(TEAM))
    return build_toolbox(
        ask_human=lambda text: approve, employees_path=employees,
        approved_path=tmp_path / "approved.json", audit_path=tmp_path / "audit.json",
    )


def test_declined_change_is_reported_by_code_without_another_model_call(tmp_path):
    toolbox = make_team_toolbox(tmp_path, approve=False)
    llm = FakeLLM([call("change_employee", action="update", name="Thandi", gross_monthly=26000), final("Are you sure?")])
    result = run_agent("set Thandi's salary to 26000", llm, toolbox)

    assert llm.step_calls == 1
    assert result.status == "approval_declined"
    assert result.message == "Cancelled. Nothing was changed."
    assert result.payslip is None


def test_approved_salary_change_is_reported_with_new_net_pay(tmp_path):
    llm = FakeLLM([call("change_employee", action="update", name="Thandi", gross_monthly=26000), final("unused")])
    result = run_agent("set Thandi's salary to 26000", llm, make_team_toolbox(tmp_path))

    assert result.status == "approved"
    assert result.message == (
        "Updated Thandi Nkosi's salary from R25,000.00 to R26,000.00. New net pay: R22,181.88 a month.")


def test_age_change_past_65_mentions_the_secondary_rebate(tmp_path):
    llm = FakeLLM([call("change_employee", action="update", name="Lerato", age=66), final("unused")])
    result = run_agent("change Lerato's age to 66", llm, make_team_toolbox(tmp_path))

    assert result.message.startswith("Updated Lerato Mokoena's age from 64 to 66.")
    assert result.message.endswith("Age 65+ adds the secondary rebate.")


def test_other_approved_changes_get_plain_confirmations(tmp_path):
    toolbox = make_team_toolbox(tmp_path)
    renamed = run_agent("rename Thandi Nkosi to Thandi Nkosie",
                        FakeLLM([call("change_employee", action="rename", name="Thandi Nkosi",
                                      new_name="Thandi Nkosie"), final()]), toolbox)
    removed = run_agent("remove Sipho Dlamini",
                        FakeLLM([call("change_employee", action="remove", name="Sipho Dlamini"), final()]), toolbox)

    assert renamed.message == "Renamed Thandi Nkosi to Thandi Nkosie."
    assert removed.message == "Removed Sipho Dlamini. Their details are kept in the audit log."


def test_a_change_that_errors_goes_to_the_model_to_explain(tmp_path):
    llm = FakeLLM([call("change_employee", action="update", name="Thandi"), final("What should change?")])
    run_agent("fix Thandi", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2


def _ask(tmp_path, question, *turns):
    return run_agent(question, FakeLLM(list(turns) or [final()]), make_team_toolbox(tmp_path))


# ---- Replies are written from the tool arguments the model chose, never from the question text ----

def test_one_figure_is_answered_in_one_sentence_with_one_model_call(tmp_path):
    llm = FakeLLM([call("employee_pay", name="Thandi", fields=["paye"]), final("unused")])
    result = run_agent("what's thandi's paye", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 1
    assert result.status == "answered"
    assert result.message == "Thandi Nkosi's PAYE is R3,381.00 a month."
    assert result.payslip.net_pay == 21441.88  # numbers come from the engine, not from the model


def test_possessive_name_and_other_figures(tmp_path):
    uif = _ask(tmp_path, "whats thandis uif", call("employee_pay", name="thandis", fields=["uif"]), final())
    both = _ask(tmp_path, "paye and uif", call("employee_pay", name="Thandi", fields=["paye", "uif"]), final())
    sdl = _ask(tmp_path, "thandis sdl", call("employee_pay", name="Thandi", fields=["sdl"]), final())
    age = _ask(tmp_path, "how old is lerato", call("employee_pay", name="Lerato", fields=["age"]), final())

    assert uif.message == "Thandi Nkosi's UIF is R177.12 a month."
    assert both.message == "Thandi Nkosi's PAYE is R3,381.00 and UIF is R177.12 a month."
    assert sdl.message == "Thandi Nkosi's SDL (paid by the employer) is R250.00 a month."
    assert age.message == "Lerato Mokoena is 64."


def test_no_fields_means_the_full_payslip(tmp_path):
    result = _ask(tmp_path, "thandi's tax numbers", call("employee_pay", name="Thandi"), final())

    assert result.message == ("For Thandi Nkosi at age 30 on R25,000.00 gross per month: "
                              "PAYE R3,381.00, UIF R177.12, net pay R21,441.88.")


def test_what_if_says_nothing_was_changed(tmp_path):
    toolbox = make_team_toolbox(tmp_path)
    result = run_agent("what would thandi's paye be at 66",
                       FakeLLM([call("employee_pay", name="Thandi", age=66, fields=["paye"]), final()]), toolbox)

    assert result.message == "At age 66, Thandi Nkosi's PAYE would be R2,567.25 a month (nothing was changed)."
    assert json.loads((tmp_path / "employees.json").read_text()) == TEAM


def test_explain_is_put_into_words_by_the_model(tmp_path):
    llm = FakeLLM([call("employee_pay", name="Thandi", explain=True), final("Because...")])
    run_agent("why is thandi's tax this amount", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2
    sent_back = [item for item in llm.history_seen_by_structure if isinstance(item, ToolResults)][0]
    assert "annual_paye" in sent_back.results[0][1]


def test_salary_and_period_conversions_match_the_tax_tables(tmp_path):
    def ask(**args):
        return _ask(tmp_path, "tax?", call("employee_pay", age=27, fields=["paye"], **args), final())

    monthly = ask(salary=50000)
    annual_from_monthly = ask(salary=50000, period="year")
    monthly_from_annual = ask(salary=600000, salary_period="year", period="month")
    annual_default = ask(salary=50000, salary_period="year")
    weekly = ask(salary=10000, salary_period="week")
    daily = ask(salary=1000, salary_period="day", days_per_week=5, period="month")

    assert monthly.message == "PAYE would be R11,075.58 a month (age 27, R50,000.00 gross a month)."
    assert annual_from_monthly.message == "PAYE would be R132,907.00 a year (age 27, R50,000.00 gross a month)."
    assert monthly_from_annual.message == "PAYE would be R11,075.58 a month (age 27, R600,000.00 gross a year)."
    assert monthly_from_annual.payslip.gross_monthly == 50000
    assert annual_default.message.endswith("a year (age 27, R50,000.00 gross a year).")
    assert weekly.message == ("PAYE would be R2,011.87 a week, averaged over 52 weeks "
                              "(age 27, R10,000.00 gross a week).")
    assert daily.message == "PAYE would be R2,514.33 a month (age 27, R1,000.00 gross a day)."
    assert daily.payslip.gross_monthly == pytest.approx(1000 * 5 * 52 / 12)


def test_anonymous_full_payslip_shows_the_inputs(tmp_path):
    result = _ask(tmp_path, "tax for a 27 year old on 50k", call("employee_pay", salary=50000, age=27), final())

    assert result.message == ("For age 27 on R50,000.00 gross per month: "
                              "PAYE R11,075.58, UIF R177.12, net pay R38,747.30.")


def test_daily_salary_without_paid_days_goes_back_to_the_model_to_ask(tmp_path):
    llm = FakeLLM(
        [call("employee_pay", salary=1000, salary_period="day", age=27), final("How many paid days per week?")],
        reply=ModelReply(status="needs_clarification", message="How many paid days per week?"),
    )
    result = run_agent("tax on R1,000 a day for a 27 year old", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2
    assert result.status == "needs_clarification"


def test_unknown_figure_goes_back_to_the_model(tmp_path):
    llm = FakeLLM([call("employee_pay", name="Thandi", fields=["wage"]), final("Which figure?")])
    run_agent("thandi's wage", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2


def test_team_questions_are_answered_from_the_arguments(tmp_path):
    def ask(answer, figure="all"):
        return _ask(tmp_path, "team?", call("team_pay", answer=answer, figure=figure), final()).message

    assert ask("count") == "There are 3 employees."
    assert ask("highest", "gross") == "Sipho Dlamini earns the most: R42,000.00 a month."
    assert ask("lowest", "gross") == "Lerato Mokoena earns the least: R8,000.00 a month."
    assert ask("highest", "paye") == "Sipho Dlamini pays the most PAYE: R8,304.75 a month."
    assert ask("lowest", "paye") == "Lerato Mokoena pays the least PAYE: R0.00 a month."
    assert ask("highest", "net") == "Sipho Dlamini takes home the most: R33,518.13 a month."
    assert ask("lowest", "net") == "Lerato Mokoena takes home the least: R7,920.00 a month."
    assert ask("highest", "sdl") == "Sipho Dlamini has the highest SDL (paid by the employer): R420.00 a month."
    assert ask("highest", "age") == "Lerato Mokoena is the oldest: 64."
    assert ask("lowest", "age") == "Thandi Nkosi is the youngest: 30."
    assert ask("total", "paye") == "For 3 employees, total monthly PAYE is R11,685.75."
    assert ask("total").startswith("Monthly payroll for 3 employees: gross R75,000.00, PAYE R11,685.75")


def test_ties_name_everyone(tmp_path):
    employees = tmp_path / "employees.json"
    employees.write_text(json.dumps([
        {"name": "Anna Smith", "age": 30, "gross_monthly": 20000, "sdl_exempt": False},
        {"name": "Ben Jones", "age": 30, "gross_monthly": 20000, "sdl_exempt": False},
        {"name": "Cara Lee", "age": 30, "gross_monthly": 10000, "sdl_exempt": False},
    ]))
    toolbox = build_toolbox(ask_human=lambda text: True, employees_path=employees,
                            approved_path=tmp_path / "approved.json", audit_path=tmp_path / "audit.json")
    result = run_agent("who earns the most",
                       FakeLLM([call("team_pay", answer="highest", figure="gross"), final()]), toolbox)

    assert result.message == "Anna Smith and Ben Jones both earn the most: R20,000.00 a month."


def test_list_answers_such_as_filters_are_left_to_the_model(tmp_path):
    llm = FakeLLM([call("team_pay", answer="list"), final("Sipho Dlamini.")])
    result = run_agent("who is over 40?", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2
    assert result.message == "ok"  # the FakeLLM's default structured reply: the model, not code, wrote it


def test_a_wrong_first_tool_is_corrected_without_a_third_model_call(tmp_path):
    llm = FakeLLM([call("team_pay", answer="total", figure="age"), call("team_pay", answer="highest", figure="paye"),
                   final()])
    result = run_agent("who pays the most paye", llm, make_team_toolbox(tmp_path))

    assert llm.step_calls == 2
    assert result.message == "Sipho Dlamini pays the most PAYE: R8,304.75 a month."


def test_read_only_questions_are_offered_only_read_only_tools(tmp_path):
    llm = FakeLLM([final("hi")])
    run_agent("Who earns the most?", llm, make_team_toolbox(tmp_path))

    assert llm.step_tool_names[0] == ["employee_pay", "team_pay"]
