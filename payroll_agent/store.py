"""Reading and writing employee data, kept separate from the tools.

Tools never open the JSON files themselves. They go through EmployeeStore, so if the
data later moves from a JSON file to Postgres, only this file needs to change.
"""

import difflib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Letters (including accented ones), spaces, apostrophes, full stops and hyphens
NAME_PATTERN = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ'.\- ]+$")


def clean_name(name) -> str:
    """Trim the ends and squash repeated spaces: '  Zanele   Mkhize ' -> 'Zanele Mkhize'."""
    return " ".join(str(name).split())


def check_name(name) -> str:
    """Return a cleaned, valid name or raise ValueError with a reason a user can understand."""
    cleaned = clean_name(name)
    if not cleaned:
        raise ValueError("Name cannot be empty.")
    if len(cleaned) > 60:
        raise ValueError("Name is too long (60 characters maximum).")
    if not NAME_PATTERN.match(cleaned):
        raise ValueError("Name can only contain letters, spaces, apostrophes, hyphens and full stops.")
    return cleaned


def _name_variants(wanted: str, exact: bool) -> list[str]:
    """The lowercased name as typed, then without a possessive ('s), then without a plain trailing s.
    exact=True (rename/remove) only drops a real possessive, never a bare s."""
    variants = [wanted]
    for suffix in ("'s", "’s"):
        if wanted.endswith(suffix) and len(wanted) > len(suffix):
            variants.append(wanted[:-len(suffix)])
            return variants
    if not exact and wanted.endswith("s") and len(wanted) > 2:
        variants.append(wanted[:-1])
    return variants


def append_record(path: Path, record: dict) -> None:
    """Add one record to a JSON list file (creating the file if needed)."""
    path = Path(path)
    records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    records.append(record)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


class EmployeeStore:
    def __init__(self, employees_path: Path, audit_path: Path):
        self.employees_path = Path(employees_path)
        self.audit_path = Path(audit_path)

    def all(self) -> list[dict]:
        return json.loads(self.employees_path.read_text(encoding="utf-8"))

    def save(self, people: list[dict]) -> None:
        self.employees_path.write_text(json.dumps(people, indent=2), encoding="utf-8")

    def log(self, record: dict) -> None:
        """Write one entry to the audit log, stamped with the time (UTC)."""
        append_record(self.audit_path, {**record, "changed_at": datetime.now(timezone.utc).isoformat()})

    def name_taken(self, name: str, ignore: str | None = None) -> bool:
        """True if someone else already has this name (ignoring case)."""
        wanted = clean_name(name).lower()
        skip = clean_name(ignore).lower() if ignore else None
        return any(p["name"].lower() == wanted and p["name"].lower() != skip for p in self.all())

    def find(self, name: str, exact: bool = False) -> dict:
        """Return one employee, or {"error": ...}.

        exact=True means the full name must match (ignoring case). Used for risky actions like
        removing or renaming, so a short name can never hit the wrong person.
        When nothing matches, the error suggests close names so typos can be caught.
        """
        wanted = clean_name(name).lower()
        people = self.all()

        for variant in _name_variants(wanted, exact):
            exact_matches = [p for p in people if p["name"].lower() == variant]
            if len(exact_matches) == 1:
                return exact_matches[0]

        if not exact:
            for variant in _name_variants(wanted, exact):
                partial = [p for p in people if variant and variant in p["name"].lower()]
                if len(partial) == 1:
                    return partial[0]
                if len(partial) > 1:
                    names = ", ".join(p["name"] for p in partial)
                    return {"error": f"More than one employee matches '{name}': {names}. Ask the user which one."}

        # Match against whole names and against each single word, so "Thandy" finds "Thandi Nkosi".
        candidates = {p["name"].lower(): [p["name"]] for p in people}
        for person in people:
            for word in person["name"].lower().split():
                candidates.setdefault(word, [])
                if person["name"] not in candidates[word]:
                    candidates[word].append(person["name"])
        close = []
        for match in difflib.get_close_matches(wanted, list(candidates), n=5, cutoff=0.6):
            close.extend(name for name in candidates[match] if name not in close)
        close = close[:3]
        kind = "with the full name" if exact else "matching"
        if close:
            return {"error": f"No employee found {kind} '{name}'. Did you mean: {', '.join(close)}? "
                             "Ask the user to confirm the right name before doing anything."}
        return {"error": f"No employee found {kind} '{name}'. Ask the user to check the name."}
