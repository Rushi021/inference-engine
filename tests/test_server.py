from fastapi.testclient import TestClient                        # in-process HTTP client

from server.app import app                                       # the naive baseline app

client = TestClient(app)                                         # spins up the app for tests


def test_health():
    r = client.get("/health")                                   # liveness check
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"                               # server is up
    assert set(body["engines"]) == {"naive", "batched"}         # both variants available


def test_generate_naive_returns_completion_and_metrics():
    r = client.post("/generate", json={"prompt": "Say hi.", "max_tokens": 16, "engine": "naive"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "naive"                            # routed to the baseline
    assert len(body["completion"].strip()) > 0                  # produced real text
    assert body["tokens_generated"] > 0                         # actually decoded tokens
    assert body["total_time_s"] > 0                             # timing captured
    assert body["tokens_per_sec"] > 0                           # throughput computed
    assert body["time_to_first_token_s"] <= body["total_time_s"] + 1e-6  # TTFT precedes completion


def test_generate_batched_engine():
    r = client.post("/generate", json={"prompt": "Say hi.", "max_tokens": 16, "engine": "batched"})
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "batched"                          # routed to continuous batching
    assert len(body["completion"].strip()) > 0                  # produced real text
    assert body["tokens_generated"] > 0                         # decoded tokens


def test_max_tokens_is_bounded():
    r = client.post("/generate", json={"prompt": "Count.", "max_tokens": 8, "engine": "naive"})
    assert r.status_code == 200
    assert r.json()["tokens_generated"] <= 8                    # respects the cap


def test_invalid_engine_rejected():
    r = client.post("/generate", json={"prompt": "Hi.", "engine": "bogus"})
    assert r.status_code == 422                                 # pydantic Literal validation
