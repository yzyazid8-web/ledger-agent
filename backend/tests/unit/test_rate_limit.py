from ledger_agent.llm.rate_limit import TokenRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_waits_when_the_window_is_full() -> None:
    clock = FakeClock()
    limiter = TokenRateLimiter(1000, clock=clock.time, sleep=clock.sleep)

    limiter.acquire(600)
    clock.now = 10.0
    limiter.acquire(600)  # 1 200 > 1 000 : attend que la première réservation expire

    assert len(clock.sleeps) == 1
    assert clock.now >= 60.0


def test_settle_frees_overestimated_tokens() -> None:
    clock = FakeClock()
    limiter = TokenRateLimiter(1000, clock=clock.time, sleep=clock.sleep)

    reservation = limiter.acquire(900)
    limiter.settle(reservation, 300)
    limiter.acquire(600)

    assert clock.sleeps == []
    assert limiter.used() == 900


def test_oversized_request_is_capped_to_the_limit() -> None:
    clock = FakeClock()
    limiter = TokenRateLimiter(1000, clock=clock.time, sleep=clock.sleep)

    limiter.acquire(5000)

    assert clock.sleeps == []
