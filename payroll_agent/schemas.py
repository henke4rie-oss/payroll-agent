"""Data shapes (Pydantic models) used by the agent.

Pydantic gives us two things:
1. Validation: bad data raises an error instead of silently passing through.
2. A schema we can hand to the model so it must answer in a fixed JSON shape.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PayslipSummary(BaseModel):
    """The numbers from a payslip. These come from the tax engine, never from the model."""

    gross_monthly: float
    paye: float
    uif_employee: float
    net_pay: float
    sdl: float = 0.0  # employer cost, not deducted from pay


class ModelReply(BaseModel):
    """What we ask the model to fill in at the end (structured output).

    Note: no numbers in here on purpose. The model writes the words,
    our code attaches the numbers.
    """

    status: Literal["answered", "needs_clarification", "approved", "approval_declined", "error"] = Field(
        description="answered = question done; needs_clarification = you need more info from the user"
    )
    message: str = Field(description="Short plain-English reply for the user")


class AgentResult(BaseModel):
    """The final result returned to the caller: the model's words plus the engine's numbers."""

    status: str
    message: str
    payslip: Optional[PayslipSummary] = None
    table: Optional[list[dict]] = None  # everyone's figures from the tools, for a UI to show beside the reply
