from greyhound_predictor.io_utils import load_card
# Импортируем после того, как вы вставите код в calculator.py
from greyhound_predictor.calculator import generate_predictions  # type: ignore[attr-defined]

def test_smoke_runs():
    card = load_card("examples/card.sample.json")
    out = generate_predictions(card)
    assert "predictions" in out and "trace" in out
