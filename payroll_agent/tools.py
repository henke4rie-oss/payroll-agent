"""The tools the agent is allowed to use.

Each tool is a plain Python function. The model never runs code itself:
it only *asks* for a tool by name, and our code decides whether to run it.

Key design choices (good things to explain in an interview):
- All tax maths happens in engine.py (exact Decimal maths). The model never calculates money.
- Anything that changes data (edit, add, rename, remove, approve) needs a human to say yes.
  That check lives in our code, not in the model's judgement.
- Tools take names and values only. Totals and payslips are recalculated by our code,
  so the model can't get a made-up number approved.
- Risky actions (rename, remove) need the full exact name, so a short name can't hit the wrong person.
- Every tool returns {"error": ...} instead of crashing, so the model can read the problem
  and ask the user instead of guessing. Not-found errors suggest close names to catch typos.
- Every approved change is written to an audit log.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Literal, Optional

from .engine import calculate_payroll, round_cents
from .schemas import PayslipSummary
from .store import EmployeeStore, append_record, check_name, clean_name

DEFAULT_EMPLOYEES = Path(__file__).parent / "employees.json"
DEFAULT_APPROVED = Path("approved_payslips.json")
DEFAULT_AUDIT = Path("audit_log.json")

# Only these fields can be changed with update_employee. Anything else is refused.
EDITABLE_FIELDS = {"gross_monthly", "age"}

# The model sees only these merged tools; each schema is resent on every call, so fewer is cheaper.
READ_ONLY_TOOLS = ("employee_pay", "team_pay")
WRITE_TOOLS = ("change_employee",)

Figure = Literal["gross", "paye", "uif", "sdl", "net", "age"]
Period = Literal["month", "year", "week", "day"]
FIGURES = ("gross", "paye", "uif", "sdl", "net", "age")


def _periods_per_year(period: str, days_per_week: int) -> Optional[int]:
    """How many of this period fit in a year. None = a daily rate without a usable days_per_week."""
    if period == "day":
        return 52 * days_per_week if 1 <= days_per_week <= 7 else None
    return {"month": 12, "year": 1, "week": 52}[period]


@dataclass
class ToolState:
    """Remembers what happened during one run so the final result can use real numbers."""

    last_payslip: Optional[PayslipSummary] = None
    last_approval: Optional[str] = None  # "approved" or "approval_declined"
    last_rows: Optional[list] = None  # everyone's figures from team_pay "list", so a UI can show them from code

    def reset(self) -> None:
        self.last_payslip = None
        self.last_approval = None
        self.last_rows = None


@dataclass
class Toolbox:
    functions: dict[str, Callable]  # internal helpers, also used directly by code and tests
    state: ToolState = field(default_factory=ToolState)
    model_tools: dict[str, Callable] = field(default_factory=dict)  # what the model may call


def _payslip(gross_monthly: float, age: int, sdl_exempt: bool) -> PayslipSummary:
    """Call the payroll engine and keep only the fields we need."""
    # str() first so Decimal gets "25000.5" and not a long binary float expansion
    result = calculate_payroll(Decimal(str(gross_monthly)), int(age), bool(sdl_exempt))
    return PayslipSummary(
        gross_monthly=float(result["gross_monthly"]),
        paye=float(result["monthly_paye"]),
        uif_employee=float(result["uif_employee"]),
        net_pay=float(result["net_pay"]),
        sdl=float(result["sdl_amount"]),
    )


def _whole_age(value) -> int:
    """Ages must be whole numbers; accept 30.0 but reject 30.5."""
    if float(value) != int(value):
        raise ValueError("Age must be a whole number.")
    return int(value)


def _positive_salary(value) -> float:
    salary = float(value)
    if salary <= 0:
        raise ValueError("Gross monthly salary must be more than zero.")
    return salary


def build_toolbox(
    ask_human: Callable[[str], bool],
    employees_path: Path = DEFAULT_EMPLOYEES,
    approved_path: Path = DEFAULT_APPROVED,
    audit_path: Path = DEFAULT_AUDIT,
) -> Toolbox:
    """Create the tools. ask_human(text) -> True/False is how a person approves things."""
    state = ToolState()
    store = EmployeeStore(employees_path, audit_path)

    def _declined(employee: str) -> dict:
        state.last_approval = "approval_declined"
        return {"status": "declined", "employee": employee}

    # NOTE: the docstrings below are read by the model to decide when to use each tool.
    # The type hints tell it what arguments to send.

    # ---------- Read-only tools ----------

    def lookup_employee(name: str) -> dict:
        """Find an employee by name (full or partial). Returns their name, age, gross monthly
        salary in Rands, and their current monthly PAYE, employee UIF and net pay.
        Returns an 'error' if the name is missing, unknown or matches several people.
        If the error suggests close names, ask the user which one they meant."""
        found = store.find(name)
        if "error" in found:
            return found
        sdl_exempt = found.get("sdl_exempt", False)
        result = {"name": found["name"], "age": found["age"], "gross_monthly": found["gross_monthly"],
                  "sdl_exempt": sdl_exempt}
        try:
            payslip = _payslip(found["gross_monthly"], found["age"], sdl_exempt)
        except (ValueError, ArithmeticError):
            return result
        state.last_payslip = payslip
        return {**result, "paye": payslip.paye, "uif_employee": payslip.uif_employee,
                "net_pay": payslip.net_pay, "sdl": payslip.sdl}

    def list_employees() -> dict:
        """List every employee with their age, gross monthly salary and monthly pay figures in Rands."""
        people = store.all()
        entries = []
        for p in people:
            entry = {"name": p["name"], "age": p["age"], "gross_monthly": p["gross_monthly"]}
            try:
                slip = _payslip(p["gross_monthly"], p["age"], p.get("sdl_exempt", False))
                entry.update(paye=slip.paye, uif_employee=slip.uif_employee, net_pay=slip.net_pay, sdl=slip.sdl)
            except (ValueError, ArithmeticError):
                pass
            entries.append(entry)
        return {"count": len(people), "employees": entries}

    def payroll_summary() -> dict:
        """Company-wide monthly totals (not per employee): gross, PAYE, employee UIF, net pay and SDL."""
        gross = paye = uif = net = sdl = Decimal("0")
        people = store.all()
        for person in people:
            slip = _payslip(person["gross_monthly"], person["age"], person.get("sdl_exempt", False))
            gross += Decimal(str(slip.gross_monthly))
            paye += Decimal(str(slip.paye))
            uif += Decimal(str(slip.uif_employee))
            net += Decimal(str(slip.net_pay))
            sdl += Decimal(str(slip.sdl))
        return {"employee_count": len(people), "total_gross": float(gross), "total_paye": float(paye),
                "total_uif_employee": float(uif), "total_net_pay": float(net), "total_sdl": float(sdl)}

    def calculate_payslip(gross_monthly: float, age: int, sdl_exempt: bool) -> dict:
        """Calculate monthly PAYE, UIF and net pay in Rands using the SARS 2026/27 tax tables.
        Use sdl_exempt=false unless the user says the employer is exempt from SDL.
        Also use this for 'what if' questions (a different age or salary) because it changes nothing.
        Returns an 'error' if the inputs are invalid."""
        try:
            payslip = _payslip(gross_monthly, age, sdl_exempt)
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}
        state.last_payslip = payslip
        return payslip.model_dump()

    def explain_payslip(gross_monthly: float, age: int, sdl_exempt: bool) -> dict:
        """Get the detailed tax breakdown behind a payslip: annual gross, each age rebate, annual and
        monthly PAYE, marginal tax rate and whether the person is below the tax threshold.
        Use this when the user asks WHY a payslip looks the way it does."""
        try:
            result = calculate_payroll(Decimal(str(gross_monthly)), int(age), bool(sdl_exempt))
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}
        keys = ["annual_gross", "primary_rebate", "secondary_rebate", "tertiary_rebate", "total_rebates",
                "annual_paye", "monthly_paye", "marginal_rate", "is_below_threshold"]
        return {k: (float(result[k]) if isinstance(result[k], Decimal) else result[k]) for k in keys}

    # ---------- Tools that change data (all need human approval) ----------

    def request_payslip_approval(employee_name: str) -> dict:
        """Ask a human to approve one employee's monthly payslip. Only call this when the user has
        explicitly asked to approve or finalise a payslip. You cannot approve it yourself: a person
        is asked, and the result says whether they approved or declined."""
        found = store.find(employee_name)
        if "error" in found:
            return found
        try:
            # We recalculate here instead of trusting numbers from the model.
            payslip = _payslip(found["gross_monthly"], found["age"], found.get("sdl_exempt", False))
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}

        summary = (
            f"Payslip for {found['name']}: gross R{payslip.gross_monthly:,.2f}, "
            f"PAYE R{payslip.paye:,.2f}, UIF R{payslip.uif_employee:,.2f}, net R{payslip.net_pay:,.2f}."
        )
        state.last_payslip = payslip

        if not ask_human(summary):
            return _declined(found["name"])

        # Only reached when a human said yes
        append_record(approved_path, {
            "employee": found["name"],
            "approved_at": datetime.now(timezone.utc).isoformat(),
            **payslip.model_dump(),
        })
        state.last_approval = "approved"
        return {"status": "approved", "employee": found["name"]}

    def update_employee(employee_name: str, field_name: str, new_value: float) -> dict:
        """Change one value for one employee. field_name must be 'gross_monthly' or 'age'.
        Only call this when the user clearly asks to change, set or update a value.
        A human is shown the before and after and must confirm. You cannot change data yourself."""
        if field_name not in EDITABLE_FIELDS:
            allowed = ", ".join(sorted(EDITABLE_FIELDS))
            return {"error": f"Cannot change '{field_name}'. Only these can be changed: {allowed}. "
                             "To change a name, rename the employee instead."}
        found = store.find(employee_name)
        if "error" in found:
            return found

        try:
            new_value = _whole_age(new_value) if field_name == "age" else _positive_salary(new_value)
            # Run the payroll engine on the new values: it rejects impossible ones (e.g. age 5)
            before = _payslip(found["gross_monthly"], found["age"], found.get("sdl_exempt", False))
            changed = {**found, field_name: new_value}
            after = _payslip(changed["gross_monthly"], changed["age"], changed.get("sdl_exempt", False))
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}

        old_value = found[field_name]
        show = (lambda v: f"R{v:,.2f}") if field_name == "gross_monthly" else str
        summary = (
            f"Change {found['name']}'s {'salary' if field_name == 'gross_monthly' else field_name}: "
            f"{show(old_value)} -> {show(new_value)}. "
            f"Net pay would go from R{before.net_pay:,.2f} to R{after.net_pay:,.2f}."
        )
        if not ask_human(summary):
            return _declined(found["name"])

        # Only reached when a human said yes: save the change, then log it
        people = store.all()
        for person in people:
            if person["name"] == found["name"]:
                person[field_name] = new_value
        store.save(people)
        store.log({"action": "update", "employee": found["name"], "field": field_name,
                   "old_value": old_value, "new_value": new_value})
        state.last_payslip = after
        state.last_approval = "approved"
        return {"status": "approved", "employee": found["name"], "field": field_name,
                "old_value": old_value, "new_value": new_value}

    def add_employee(name: str, age: int, gross_monthly: float) -> dict:
        """Add a new employee. Only call this when the user clearly asks to add or hire someone AND has
        given the name, age and gross monthly salary in Rands. If any of these is missing, ask the user.
        A human must confirm."""
        try:
            clean = check_name(name)
            if store.name_taken(clean):
                raise ValueError(f"An employee called '{clean}' already exists.")
            age = _whole_age(age)
            gross = _positive_salary(gross_monthly)
            payslip = _payslip(gross, age, False)
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}

        summary = f"Add new employee {clean}: age {age}, gross R{gross:,.2f} (net pay R{payslip.net_pay:,.2f})."
        if not ask_human(summary):
            return _declined(clean)

        people = store.all()
        people.append({"name": clean, "age": age, "gross_monthly": gross, "sdl_exempt": False})
        store.save(people)
        store.log({"action": "add", "employee": clean, "age": age, "gross_monthly": gross})
        state.last_payslip = payslip
        state.last_approval = "approved"
        return {"status": "approved", "employee": clean}

    def rename_employee(employee_name: str, new_name: str) -> dict:
        """Change an employee's name, for example to fix a typo. Needs the employee's current FULL name
        and the new full name. A human must confirm. You cannot rename anyone yourself."""
        found = store.find(employee_name, exact=True)
        if "error" in found:
            return found
        try:
            clean = check_name(new_name)
            if clean == found["name"]:
                raise ValueError("The new name is the same as the current name.")
            if store.name_taken(clean, ignore=found["name"]):
                raise ValueError(f"Another employee is already called '{clean}'.")
        except ValueError as err:
            return {"error": str(err)}

        if not ask_human(f"Rename '{found['name']}' to '{clean}'."):
            return _declined(found["name"])

        people = store.all()
        for person in people:
            if person["name"] == found["name"]:
                person["name"] = clean
        store.save(people)
        store.log({"action": "rename", "old_name": found["name"], "new_name": clean})
        state.last_approval = "approved"
        return {"status": "approved", "old_name": found["name"], "new_name": clean}

    def remove_employee(employee_name: str) -> dict:
        """Remove an employee. Needs their FULL name. Only call this when the user clearly asks to
        remove or delete someone. A human must confirm. The details are kept in the audit log."""
        found = store.find(employee_name, exact=True)
        if "error" in found:
            return found

        summary = (f"Remove {found['name']} (age {found['age']}, gross R{found['gross_monthly']:,.2f}). "
                   "Their details are kept in the audit log.")
        if not ask_human(summary):
            return _declined(found["name"])

        store.save([p for p in store.all() if p["name"] != found["name"]])
        store.log({"action": "remove", "employee": found["name"], "removed_record": found})
        state.last_approval = "approved"
        return {"status": "approved", "employee": found["name"]}

    functions = {
        "lookup_employee": lookup_employee,
        "list_employees": list_employees,
        "payroll_summary": payroll_summary,
        "calculate_payslip": calculate_payslip,
        "explain_payslip": explain_payslip,
        "request_payslip_approval": request_payslip_approval,
        "update_employee": update_employee,
        "add_employee": add_employee,
        "rename_employee": rename_employee,
        "remove_employee": remove_employee,
    }

    # ---------- Tools the model sees (docstrings are sent to the model: keep them short) ----------

    def employee_pay(name: str = "", fields: list[Figure] = None, age: int = 0, salary: float = 0,
                     salary_period: Period = "month", period: Period = "", days_per_week: int = 0,
                     explain: bool = False) -> dict:
        """Pay for ONE employee (by name) or for a given salary and age. fields: only the figures asked for
        (gross, paye, uif, sdl, net, age); empty = full payslip. A name plus age or salary is a what-if
        (nothing is saved). salary_period: how the salary is quoted; period: how to show the answer (default
        same as salary_period). Daily needs days_per_week. explain=true only when asked WHY tax is what it is."""
        fields = [fields] if isinstance(fields, str) else list(fields or [])
        unknown = [f for f in fields if f not in FIGURES]
        if unknown:
            return {"error": f"Unknown figure(s) {unknown}. Use: {', '.join(FIGURES)}."}
        person = {}
        if clean_name(name):
            person = store.find(name)
            if "error" in person:
                return person

        years = age or person.get("age", 0)
        if salary:
            if _periods_per_year(salary_period, days_per_week) is None:
                return {"error": "Need days_per_week (paid workdays per week, 1 to 7) for a daily rate. Ask the user."}
            if salary_period == "month":
                gross_monthly = salary
            elif salary_period == "year":
                gross_monthly = salary / 12
            elif salary_period == "week":
                gross_monthly = salary * 52 / 12
            else:
                gross_monthly = salary * days_per_week * 52 / 12
        else:
            gross_monthly = person.get("gross_monthly", 0)
        if not gross_monthly or not years:
            return {"error": "Need an employee name, or both salary and age. Ask the user."}

        shown = period or (salary_period if salary else "month")
        shown_per_year = _periods_per_year(shown, days_per_week)
        if shown_per_year is None:
            return {"error": "Need days_per_week (paid workdays per week, 1 to 7) for a daily rate. Ask the user."}
        exempt = bool(person.get("sdl_exempt", False))
        try:
            monthly = _payslip(gross_monthly, years, exempt)
            calc = calculate_payroll(Decimal(str(gross_monthly)), int(years), exempt)
        except (ValueError, ArithmeticError) as err:
            return {"error": str(err)}

        if shown == "month":
            gross, paye, uif, sdl = calc["gross_monthly"], calc["monthly_paye"], calc["uif_employee"], calc["sdl_amount"]
            net = calc["net_pay"]
        else:
            n = Decimal(shown_per_year)
            gross, paye = calc["annual_gross"] / n, calc["annual_paye"] / n
            uif, sdl = calc["uif_employee"] * 12 / n, calc["sdl_amount"] * 12 / n
            net = gross - paye - uif
        figures = {"gross": gross, "paye": paye, "uif": uif, "sdl": sdl, "net": net}
        result = {"age": years, "period": shown, "fields": fields,
                  **{key: float(round_cents(value)) for key, value in figures.items()}}
        if person:
            result["name"] = person["name"]
            result["what_if"] = bool(age or salary)
            result["age_override"] = bool(age)
        if salary:
            result["salary"] = salary
            result["salary_period"] = salary_period
        if days_per_week:
            result["days_per_week"] = days_per_week
        if explain:
            result.update(explain_payslip(gross_monthly, years, exempt))
        state.last_payslip = monthly
        return result

    def team_pay(answer: Literal["list", "count", "highest", "lowest", "total"] = "list",
                 figure: Literal["gross", "paye", "uif", "sdl", "net", "age", "all"] = "all") -> dict:
        """Pay across ALL employees. answer: list (everyone's figures; use for filters like 'over 60'),
        count, highest or lowest (who has the most or least of figure), total (company total of figure;
        'all' = every figure)."""
        rows = []
        for p in list_employees()["employees"]:
            row = {"name": p["name"], "age": p["age"], "gross": p["gross_monthly"]}
            for key, source in (("paye", "paye"), ("uif", "uif_employee"), ("sdl", "sdl"), ("net", "net_pay")):
                if source in p:
                    row[key] = p[source]
            rows.append(row)
        if answer == "count":
            return {"answer": "count", "count": len(rows)}
        if answer == "list":
            state.last_rows = rows
            return {"answer": "list", "count": len(rows), "employees": rows}
        if answer in ("highest", "lowest"):
            figure = "gross" if figure == "all" else figure
            pool = [row for row in rows if figure in row]
            if not pool:
                return {"error": "There are no employees to compare."}
            best = (max if answer == "highest" else min)(row[figure] for row in pool)
            return {"answer": answer, "figure": figure, "value": best,
                    "names": [row["name"] for row in pool if row[figure] == best]}
        if answer == "total":
            if figure == "age":
                return {"error": "A total of ages isn't useful. Use list, highest or lowest instead."}
            keys = [k for k in FIGURES if k != "age"] if figure == "all" else [figure]
            totals = {k: float(round_cents(sum(Decimal(str(row.get(k, 0))) for row in rows))) for k in keys}
            return {"answer": "total", "figure": figure, "count": len(rows), "totals": totals}
        return {"error": f"Unknown answer '{answer}'. Use list, count, highest, lowest or total."}

    def change_employee(action: Literal["approve", "update", "add", "rename", "remove"], name: str,
                        age: int = 0, gross_monthly: float = 0, new_name: str = "") -> dict:
        """Change data; a person must confirm. approve: an employee's payslip. update: change an EXISTING
        employee's age or salary (salary = gross_monthly); give exactly one. add: a NEW employee, needs name,
        age and gross_monthly. rename: needs new_name. rename/remove need the exact full name."""
        if action == "approve":
            return request_payslip_approval(name)
        if action == "update":
            given = [(f, v) for f, v in (("age", age), ("gross_monthly", gross_monthly)) if v]
            if len(given) != 1:
                return {"error": "For update give exactly one of age or gross_monthly."}
            return update_employee(name, *given[0])
        if action == "add":
            if not (age and gross_monthly):
                return {"error": "To add an employee I need their name, age and gross monthly salary. Ask the user."}
            return add_employee(name, age, gross_monthly)
        if action == "rename":
            if not clean_name(new_name):
                return {"error": "rename needs new_name."}
            return rename_employee(name, new_name)
        if action == "remove":
            return remove_employee(name)
        return {"error": f"Unknown action '{action}'. Use approve, update, add, rename or remove."}

    model_tools = {
        "employee_pay": employee_pay,
        "team_pay": team_pay,
        "change_employee": change_employee,
    }
    return Toolbox(functions=functions, state=state, model_tools=model_tools)
