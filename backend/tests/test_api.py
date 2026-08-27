"""End-to-end API tests against the mock backend and a throwaway SQLite DB."""
import json

RESUME = "Backend engineer, 5 years Python, FastAPI, PostgreSQL, Docker, AWS, CI/CD."
JD = "Backend Engineer. 3+ years Python, FastAPI, PostgreSQL, Docker, cloud, testing."

PAIR = {"resume": RESUME, "job_description": JD}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_backend": "mock"}


def test_compare_returns_both_models_and_delta(client):
    r = client.post("/compare", json=PAIR)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"base", "finetuned", "score_delta"}
    assert 0 <= body["base"]["score"] <= 100
    assert 0 <= body["finetuned"]["score"] <= 100
    assert body["score_delta"] == body["finetuned"]["score"] - body["base"]["score"]


def test_compare_validates_empty_input(client):
    r = client.post("/compare", json={"resume": "", "job_description": JD})
    assert r.status_code == 422


def test_score_streams_sse_and_finishes_with_done(client):
    with client.stream("POST", "/score", json=PAIR) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data:"):])
            for line in r.iter_lines()
            if line.startswith("data:")
        ]
    assert events[-1]["type"] == "done"
    assert 0 <= events[-1]["score"] <= 100
    assert any(e["type"] == "chunk" for e in events)


def test_compare_then_history_records_the_request(client):
    client.post("/compare", json=PAIR)
    r = client.get("/history?page=1&page_size=10")
    assert r.status_code == 200
    page = r.json()
    assert page["total"] == 1
    row = page["items"][0]
    assert row["jd_text"] == JD
    assert row["is_comparison"] is True
    assert row["base_score"] is not None


def test_history_pagination_params(client):
    r = client.get("/history?page=1&page_size=0")
    assert r.status_code == 422  # page_size must be >= 1


def test_benchmark_404_without_results(client):
    r = client.get("/eval/benchmark")
    assert r.status_code == 404


def test_dataset_stats_degrades_gracefully(client):
    r = client.get("/datasets/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["total"] == 0
    assert "labeler" in body
