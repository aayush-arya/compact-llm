"""Unit tests for the mock inference engine and the shared output parser."""
from app.models.inference import MockEngine, parse_output

STRONG_RESUME = (
    "Senior backend engineer, 6 years Python Django FastAPI PostgreSQL AWS Docker "
    "Kubernetes CI/CD microservices. Led a team, ran on-call, mentored engineers."
)
STRONG_JD = (
    "Senior Backend Engineer. 4+ years Python, Django or FastAPI, PostgreSQL, AWS, "
    "Docker, Kubernetes, CI/CD, microservices, technical leadership, mentoring."
)
WEAK_RESUME = "Executive chef, 14 years. Menu planning, food cost control, vendor negotiation."
WEAK_JD = "Senior iOS Engineer. 8+ years Swift, Objective-C, UIKit, SwiftUI, Core Data."


def test_parse_output_reads_score_and_rationale():
    score, rationale = parse_output("Score: 73\nRationale: Solid domain overlap.")
    assert score == 73
    assert rationale == "Solid domain overlap."


def test_parse_output_clamps_and_defaults():
    assert parse_output("Score: 250\nRationale: x")[0] == 100
    assert parse_output("no score here")[0] == 50


def test_mock_is_deterministic():
    a = MockEngine().generate(STRONG_RESUME, STRONG_JD, use_adapter=True)
    b = MockEngine().generate(STRONG_RESUME, STRONG_JD, use_adapter=True)
    assert a.score == b.score


def test_mock_strong_match_scores_higher_than_weak_match():
    eng = MockEngine()
    strong = eng.generate(STRONG_RESUME, STRONG_JD, use_adapter=True).score
    weak = eng.generate(WEAK_RESUME, WEAK_JD, use_adapter=True).score
    assert strong > 70
    assert weak < 30
    assert strong > weak


def test_mock_finetuned_is_more_decisive_than_base():
    """The fine-tuned path commits; the base path hedges toward 50."""
    eng = MockEngine()
    ft = eng.generate(STRONG_RESUME, STRONG_JD, use_adapter=True).score
    base = eng.generate(STRONG_RESUME, STRONG_JD, use_adapter=False).score
    assert abs(ft - 50) > abs(base - 50)


def test_mock_finetuned_path_is_faster():
    eng = MockEngine()
    ft = eng.generate(STRONG_RESUME, STRONG_JD, use_adapter=True).latency_ms
    base = eng.generate(STRONG_RESUME, STRONG_JD, use_adapter=False).latency_ms
    assert ft < base


def test_mock_stream_reconstructs_to_parseable_text():
    chunks = list(MockEngine().generate_stream(STRONG_RESUME, STRONG_JD, use_adapter=True))
    score, rationale = parse_output("".join(chunks))
    assert 0 <= score <= 100
    assert rationale
