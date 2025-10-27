from greyhound_predictor.io_utils import load_card
from greyhound_predictor.calculator import generate_predictions

def test_contract_sample():
    card = load_card("examples/card.sample.json")
    res = generate_predictions(card)

    assert "error" not in res
    preds = [p for p in res["predictions"] if p["rank"] is not None]

    total = round(sum(p["percentage"] for p in preds), 1)
    assert abs(total - 100.0) <= 0.1

    ranks = [p["rank"] for p in preds]
    assert ranks == sorted(ranks)
    assert len(set(ranks)) == len(ranks)

    trace = res["trace"]
    assert trace["run_mode"] in {"Time", "Rank"}
    assert len(trace["trace_hash"]) == 64
