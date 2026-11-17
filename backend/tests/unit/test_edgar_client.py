import httpx
import pytest

from ledger_agent.ingestion.edgar_client import EdgarClient, EdgarError, RateLimiter

USER_AGENT = "ledger-agent-tests test@example.com"


def make_client(handler: httpx.MockTransport, sleeps: list[float]) -> EdgarClient:
    return EdgarClient(
        USER_AGENT,
        max_requests_per_second=1000,
        max_retries=3,
        transport=handler,
        sleep=sleeps.append,
    )


def test_user_agent_must_contain_an_email() -> None:
    with pytest.raises(ValueError, match="contact email"):
        EdgarClient("ledger-agent")


def test_user_agent_header_is_sent() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["User-Agent"])
        return httpx.Response(200, json={"cik": 66740})

    with make_client(httpx.MockTransport(handler), []) as client:
        assert client.company_facts(66740) == {"cik": 66740}

    assert seen == [USER_AGENT]


def test_company_facts_url_pads_cik() -> None:
    assert EdgarClient.company_facts_url(66740).endswith("/CIK0000066740.json")


def test_retries_on_429_then_succeeds() -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "2"}),
            httpx.Response(503),
            httpx.Response(200, content=b"ok"),
        ]
    )
    sleeps: list[float] = []

    with make_client(httpx.MockTransport(lambda _: next(responses)), sleeps) as client:
        assert client.get_bytes("https://data.sec.gov/x") == b"ok"

    # Retry-After respecté, puis attente exponentielle (2**1)
    assert [s for s in sleeps if s >= 1] == [2.0, 2.0]


def test_gives_up_after_max_retries() -> None:
    with (
        make_client(httpx.MockTransport(lambda _: httpx.Response(503)), []) as client,
        pytest.raises(EdgarError, match="503"),
    ):
        client.get_bytes("https://data.sec.gov/x")


def test_403_gives_an_actionable_error() -> None:
    with (
        make_client(httpx.MockTransport(lambda _: httpx.Response(403)), []) as client,
        pytest.raises(EdgarError, match="SEC_USER_AGENT"),
    ):
        client.get_bytes("https://data.sec.gov/x")


def test_rate_limiter_spaces_calls() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(4, clock=lambda: now[0], sleep=fake_sleep)
    for _ in range(3):
        limiter.wait()

    assert sleeps == [0.25, 0.25]
