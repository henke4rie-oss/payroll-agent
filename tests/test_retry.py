"""Tests for the retry logic. No network: we fake the API call and the sleeping."""

import pytest

from payroll_agent.llm import MAX_ATTEMPTS, with_retry


class APIError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code
        super().__init__(f"API returned {status_code}")


def api_error(code):
    return APIError(code)


def flaky(failures, code):
    """A fake API call that fails `failures` times, then works."""
    state = {"calls": 0}

    def make_call():
        state["calls"] += 1
        if state["calls"] <= failures:
            raise api_error(code)
        return "ok"

    return make_call, state


def test_rate_limit_is_retried_with_growing_waits():
    call, state = flaky(2, 429)
    waits = []
    assert with_retry(call, sleep=waits.append) == "ok"
    assert state["calls"] == 3
    assert waits == [2, 4]


def test_gives_up_after_the_maximum_attempts():
    call, state = flaky(99, 503)
    with pytest.raises(APIError):
        with_retry(call, sleep=lambda s: None)
    assert state["calls"] == MAX_ATTEMPTS


def test_bad_key_or_bad_request_is_not_retried():
    for code in (400, 401, 403, 404):
        call, state = flaky(99, code)
        with pytest.raises(APIError):
            with_retry(call, sleep=lambda s: None)
        assert state["calls"] == 1, code


def test_success_first_time_never_sleeps():
    waits = []
    assert with_retry(lambda: "ok", sleep=waits.append) == "ok"
    assert waits == []


def test_timeouts_and_connection_errors_are_retried():
    from openai import APITimeoutError

    state = {"calls": 0}

    def make_call():
        state["calls"] += 1
        if state["calls"] == 1:
            raise APITimeoutError(request=object())
        return "ok"

    waits = []
    assert with_retry(make_call, sleep=waits.append) == "ok"
    assert state["calls"] == 2
    assert waits == [2]


def test_quota_exhaustion_is_not_retried():
    error = APIError(429)
    error.body = {"error": {"code": "insufficient_quota", "message": "quota exceeded"}}
    calls = {"count": 0}

    def fail_with_quota():
        calls["count"] += 1
        raise error

    waits = []
    with pytest.raises(APIError):
        with_retry(fail_with_quota, sleep=waits.append)

    assert calls["count"] == 1
    assert waits == []
