# src/greyhound_predictor/io_utils.py
import json
from pathlib import Path

NBSP = "\u00A0"

def load_card(path: str | Path) -> dict:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    clean = txt.replace(NBSP, " ")
    return json.loads(clean)

def load_cards(path: str | Path) -> list[dict]:
    """Поддерживает один объект {…} или массив объектов [{…}, {…}, …]."""
    p = Path(path)
    txt = p.read_text(encoding="utf-8").replace(NBSP, " ")
    data = json.loads(txt)
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    raise ValueError("Файл должен содержать объект или массив объектов JSON")

def load_jsonl_cards(path: str | Path) -> list[dict]:
    """Читает JSON Lines (по одному объекту в строке). Пустые строки/строки с # пропускает."""
    p = Path(path)
    out: list[dict] = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(json.loads(line.replace(NBSP, " ")))
    return out

def save_json(obj, path: str | Path, pretty: bool = True) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False),
        encoding="utf-8",
    )
