from fastapi.testclient import TestClient                        # in-process HTTP client

from server.app import app                                       # the naive baseline app

client = TestClient(app)                                         # spins up the app for tests


def test_health():
    r = client.get("/health")                                   # liveness check
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"                               # server is up
    assert body["engine"] == "naive"                            # serving the baseline engine


def test_generate_returns_completion_and_metrics():
    r = client.post("/generate", json={"prompt": "Say hi.", "max_tokens": 16})
    assert r.status_code == 200
    body = r.json()
    assert len(body["completion"].strip()) > 0                  # produced real text
    assert body["tokens_generated"] > 0                         # actually decoded tokens
    assert body["total_time_s"] > 0                             # timing captured
    assert body["tokens_per_sec"] > 0                           # throughput computed
    assert body["time_to_first_token_s"] <= body["total_time_s"] + 1e-6  # TTFT precedes completion


def test_max_tokens_is_bounded():
    r = client.post("/generate", json={"prompt": "Count.", "max_tokens": 8})
    assert r.status_code == 200
    assert r.json()["tokens_generated"] <= 8                    # respects the cap
