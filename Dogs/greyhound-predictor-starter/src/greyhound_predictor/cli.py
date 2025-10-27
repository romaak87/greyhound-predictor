# src/greyhound_predictor/cli.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from greyhound_predictor.io_utils import load_card, load_cards, load_jsonl_cards, save_json
from greyhound_predictor.calculator import generate_predictions
from greyhound_predictor import __version__

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="greyhound-predictor",
        description="OFF-NET v1.3 — генератор прогнозов по карточкам забегов",
    )
    ap.add_argument("inputs", nargs="+", help="card.json / cards.json / cards.jsonl или папка")
    ap.add_argument("-o", "--out", default="-", help="файл для результата или '-' для stdout")
    ap.add_argument("--pretty", action="store_true", help="красивое форматирование JSON")
    args = ap.parse_args()

    # собираем файлы
    files: list[Path] = []
    for arg in args.inputs:
        p = Path(arg)
        if p.is_dir():
            files.extend(sorted(list(p.glob("*.json")) + list(p.glob("*.jsonl")) + list(p.glob("*.ndjson"))))
        else:
            files.append(p)

    results = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            cards = load_jsonl_cards(path)
        else:
            cards = load_cards(path)  # поддерживает и одиночный объект, и массив

        for idx, card in enumerate(cards):
            res = generate_predictions(card)
            results.append({"file": str(path), "index": idx, **res})

    payload = results[0] if len(results) == 1 else {"batch": results, "version": __version__}

    if args.out == "-":
        print(json.dumps(payload, indent=2 if args.pretty else None, ensure_ascii=False))
    else:
        save_json(payload, args.out, pretty=args.pretty)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
