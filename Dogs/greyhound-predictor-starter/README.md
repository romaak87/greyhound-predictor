# greyhound-predictor (starter)

Минимальный каркас под ваш OFF-NET v1.3.

## Шаги
1) Активируйте venv и установите зависимости:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip -r requirements.txt
   ```
2) Вставьте ваш `my_calculator.py (v1.3)` в `src/greyhound_predictor/calculator.py` (уберите блок __main__).
3) Запуск:
   ```bash
   python -m greyhound_predictor.cli examples/card.sample.json --pretty
   ```
4) Тесты:
   ```bash
   pytest
   ```

