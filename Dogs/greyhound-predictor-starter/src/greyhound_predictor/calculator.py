# --- ФИНАЛЬНЫЙ КОД: my_calculator.py (v1.3 - Исправлен баг 5.x < 0, улучшен парсинг) ---
from __future__ import annotations
import re
import math
import json
import hashlib
from typing import Optional, Dict, List, Any, Tuple
from statistics import median, StatisticsError
from collections import Counter
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

# --- D0: КОНСТАНТЫ ТОЧНОСТИ ---
EPSILON = 1e-9

def is_le(a: float, b: float) -> bool:
    return a <= (b + EPSILON)

def is_ge(a: float, b: float) -> bool:
    return a >= (b - EPSILON)

def is_eq(a: float, b: float) -> bool:
    return abs(a - b) < EPSILON

# --- D1.1: НОРМАЛИЗАЦИЯ КОММЕНТАРИЕВ (v1.1) ---
def normalize_comment(comment: str) -> str:
    if not comment:
        return ""
    text = comment.lower().strip()
    text = re.sub(r"[.,;/:\\_—–\-]+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\b(st|nd|rd|th)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- D1.2: СЛОВАРЬ ТОКЕНОВ (v1.1, исправлен дубль ld) ---
TOKEN_REGEXES = {
    'qaw': r'\b(q ?away|quick ?away|vqaw|vq ?aw|q ?aw)\b',
    'ep': r'\bep\b',
    # (Патч v1.3: Убран дубль 'ld')
    'led': r'\b(led|always led|ld|soon led|sn led|soonled|ald)\b',
    'saw': r'\b(s ?aw|slow ?away|vsaw|msd brk|stumbled start)\b',
    'trouble': r'\b(crowded|crd|bumped|bmp|blk|baulked|forced to check|fcd ?(?:to )?ck|struck(?:into)?)\b',
    'trouble_soft':  r'\b(crowded|crd)\b',
    'trouble_heavy': r'\b(bumped|bmp|blk|baulked|forced to check|fcd ?(?:to )?ck|struck(?:into)?)\b',
    'finisher': r'\b(ran on|fin well|fin strong|held on|ev ch|chl)\b',
    'rails': r'\b(rails|rls)\b',
    'mid': r'\b(mid|middle)\b',
    'wide': r'\b(wide|vwide|w)\b',
}

def scan_tokens(normalized_comment: str) -> Dict[str, bool]:
    found_tokens = {name: False for name in TOKEN_REGEXES.keys()}
    if not normalized_comment:
        return found_tokens
    for token_name, regex_pattern in TOKEN_REGEXES.items():
        if re.search(regex_pattern, normalized_comment):
            found_tokens[token_name] = True
    return found_tokens

# --- D1.4: ПАРСИНГ bndPos (v1.2) ---
def parse_bndpos(bndpos_str: Optional[str]) -> Optional[int]:
    if not bndpos_str:
        return None
    s = bndpos_str.lower().strip()
    m = re.match(r'^\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*(?:st|nd|rd|th)?\s*$', s)
    if m:
        return int(m.group(2))
    m = re.match(r'^\s*(\d)\s*', s)
    if m:
        return int(m.group(1))
    return None

# --- (Патч v1.3: Помощник для парсинга чисел) ---
def to_float(x: Any) -> Optional[float]:
    """Преобразует значение в float, если возможно."""
    if x is None: return None
    try:
        # Убираем пробелы и преобразуем
        return float(str(x).strip())
    except (ValueError, TypeError):
        return None

# --- D1.8: ПАРСИНГ calc_time ---
def calculate_calc_time(result_run_time: Optional[float], going_type: Optional[float]) -> Optional[float]:
    # (Используем to_float для большей надежности)
    rt = to_float(result_run_time)
    gt = to_float(going_type)

    if rt is None or is_le(rt, 0.0):
        return None

    going_type_val = gt if gt is not None else 0.0
    return rt - (going_type_val / 100.0)

# --- "Безопасная" Медиана (D3) ---
def safe_median(data: List[float]) -> Optional[float]:
    if not data:
        return None
    try:
        # Убедимся, что все элементы - float
        float_data = [d for d in data if isinstance(d, (int, float))]
        if not float_data: return None
        return median(float_data)
    except StatisticsError:
        return None

# --- ШАГ 4: СБОРКА "СКЕЛЕТА" (КОНТЕЙНЕРЫ) ---
@dataclass
class ProcessedForm:
    index: int
    race_class: str
    distance: int
    calc_time: Optional[float]
    valid_position: Optional[int]
    valid_sectional: Optional[float]
    bnd_pos: Optional[int]
    tokens: Dict[str, bool]

@dataclass
class ProcessedDog:
    dog_name: str
    trap_number: int
    forms: List[ProcessedForm]
    style: str = "mid"
    front_rate_raw: float = 0.0
    neg_start_rate_raw: float = 0.0
    trouble_rate_raw: float = 0.0
    front_rate_adj: float = 0.0
    neg_start_rate_adj: float = 0.0
    trouble_rate_adj: float = 0.0
    finisher_flag: bool = False
    neg_start_flag: bool = False
    n_valid_l5: int = 0
    dog_median_class: Optional[float] = None
    cbt: Optional[float] = None
    bda: Optional[float] = None
    cbt_source: Optional[str] = None
    bda_mode: Optional[str] = None
    early_dog_median_split: Optional[float] = None
    late_pace_rw: Optional[float] = None
    raw_score: float = 0.0
    flags: List[str] = field(default_factory=list)
    mods_trace: Dict[str, float] = field(default_factory=dict)
    trouble_cap_total_trace: float = 0.0
    bonus_cap_total_trace: float = 0.0
    neg_flags_count: int = 0
    pos_flags_count: int = 0
    percentage: float = 0.0
    rank: Optional[int] = None
    base_score_trace: float = 0.0

# --- ШАГ 4: Главный "Парсер Карточки" (ИСПРАВЛЕНО v1.3) ---
def process_card(race_card: Dict[str, Any]) -> List[ProcessedDog]:
    processed_dogs_list: List[ProcessedDog] = []

    for dog_data in race_card.get('dogs', []):
        processed_forms_list: List[ProcessedForm] = []

        forms_data = dog_data.get('form', {}).get('forms', [])
        for index, form_data in enumerate(forms_data):

            norm_comment = normalize_comment(form_data.get('raceComment'))
            tokens = scan_tokens(norm_comment)
            bnd_pos = parse_bndpos(form_data.get('bndPos'))
            calc_time = calculate_calc_time(
                form_data.get('resultRunTime'),
                form_data.get('goingType')
            )

            # --- (Патч v1.3: Улучшен парсинг pos/sect) ---
            valid_position: Optional[int] = None
            pos_raw = form_data.get('resultPosition')
            if pos_raw is not None:
                pos_float = to_float(pos_raw)
                if pos_float is not None:
                    try:
                        p = int(pos_float)
                        if 1 <= p <= 6:
                            valid_position = p
                    except (ValueError, TypeError):
                        pass # Останется None

            valid_sectional: Optional[float] = None
            sect_raw = form_data.get('sectionalTime')
            s = to_float(sect_raw)
            if s is not None and is_ge(s, 0.001):
                valid_sectional = s
            # --- Конец патча v1.3 ---

            # Используем to_float и для distance для надежности
            dist_val = to_float(form_data.get('distance'))

            processed_form = ProcessedForm(
                index=index,
                race_class=str(form_data.get('raceClass', 'UNKNOWN')),
                distance=int(dist_val) if dist_val is not None else 0,
                calc_time=calc_time,
                valid_position=valid_position,
                valid_sectional=valid_sectional,
                bnd_pos=bnd_pos,
                tokens=tokens
            )
            processed_forms_list.append(processed_form)

        processed_dog = ProcessedDog(
            dog_name=str(dog_data.get('dogName', 'UNKNOWN_DOG')),
            trap_number=int(to_float(dog_data.get('trapNumber')) or 0),
            forms=processed_forms_list
        )
        processed_dogs_list.append(processed_dog)

    return processed_dogs_list

# --- ШАГ 5: Агрегаты (п. 2.3) (v1.1) ---
def calculate_dog_aggregates(dog: ProcessedDog, field_median_rates: Dict[str, float]):
    l5_forms = dog.forms[:5]
    n_total = len(l5_forms)
    dog.n_valid_l5 = n_total

    if n_total == 0:
        return

    style_slots = []
    counts = {'front': 0, 'neg_start': 0, 'trouble': 0, 'finisher': 0}

    for i, form in enumerate(l5_forms):
        if i == 0 and form.tokens['saw']:
            dog.neg_start_flag = True

        style_slot = None
        if form.tokens['rails']: style_slot = 'rails'
        elif form.tokens['wide']: style_slot = 'wide'
        elif form.tokens['mid']: style_slot = 'mid'
        if style_slot: style_slots.append(style_slot)

        if form.tokens['qaw'] or form.tokens['ep'] or form.tokens['led']: counts['front'] += 1
        if form.tokens['saw']: counts['neg_start'] += 1
        if form.tokens['trouble']: counts['trouble'] += 1
        if form.tokens['finisher']: counts['finisher'] += 1

    if style_slots:
        style_counts = Counter(style_slots).most_common()
        if len(style_counts) > 1 and style_counts[0][1] == style_counts[1][1]:
            styles_in_list = set(style_slots)
            if 'rails' in styles_in_list: dog.style = 'rails'
            elif 'wide' in styles_in_list: dog.style = 'wide'
            else: dog.style = 'mid'
        else:
            dog.style = style_counts[0][0]

    dog.front_rate_raw = counts['front'] / n_total if n_total > 0 else 0.0
    dog.neg_start_rate_raw = counts['neg_start'] / n_total if n_total > 0 else 0.0
    dog.trouble_rate_raw = counts['trouble'] / n_total if n_total > 0 else 0.0

    dog.front_rate_adj = dog.front_rate_raw
    dog.neg_start_rate_adj = dog.neg_start_rate_raw
    dog.trouble_rate_adj = dog.trouble_rate_raw

    if 3 <= n_total <= 4:
        weight_dog = n_total / 5.0
        weight_field = 1.0 - weight_dog
        dog.front_rate_adj = (weight_dog * dog.front_rate_raw) + (weight_field * field_median_rates['front'])
        dog.neg_start_rate_adj = (weight_dog * dog.neg_start_rate_raw) + (weight_field * field_median_rates['neg_start'])
        dog.trouble_rate_adj = (weight_dog * dog.trouble_rate_raw) + (weight_field * field_median_rates['trouble'])

    if counts['finisher'] >= 2:
        dog.finisher_flag = True

    if dog.neg_start_flag and "NEG_START" not in dog.flags:
        dog.flags.append("NEG_START")

def process_aggregates(all_dogs: List[ProcessedDog]):
    field_rates = {'front': [], 'neg_start': [], 'trouble': []}
    empty_meds = {'front': 0.0, 'neg_start': 0.0, 'trouble': 0.0}

    for dog in all_dogs:
        calculate_dog_aggregates(dog, empty_meds)
        if dog.n_valid_l5 >= 1:
            field_rates['front'].append(dog.front_rate_raw)
            field_rates['neg_start'].append(dog.neg_start_rate_raw)
            field_rates['trouble'].append(dog.trouble_rate_raw)

    field_median_rates = {
        'front': safe_median(field_rates['front']) or 0.0,
        'neg_start': safe_median(field_rates['neg_start']) or 0.0,
        'trouble': safe_median(field_rates['trouble']) or 0.0,
    }

    for dog in all_dogs:
        calculate_dog_aggregates(dog, field_median_rates)

# --- ШАГ 6: CBT и BDA (п 3.1) ---
def calculate_cbt_bda(dog: ProcessedDog, race_distance: int, race_grade: str):
    dist_min = race_distance * 0.8
    dist_max = race_distance * 1.2
    s_cur_times, s_adj_calcs = [], []
    s_cur_l5_times, s_adj_l5_calcs = [], []
    s_cur_forms_all = []

    for form in dog.forms:
        if form.calc_time is None: continue
        # Используем to_float для distance в проверках
        form_dist_float = to_float(form.distance)
        if form_dist_float is None: continue

        if is_eq(form_dist_float, race_distance):
            s_cur_times.append(form.calc_time)
            s_cur_forms_all.append(form)
            if form.index < 5: s_cur_l5_times.append(form.calc_time)
        elif is_ge(form_dist_float, dist_min) and is_le(form_dist_float, dist_max):
            # Защита от деления на ноль, хотя distance вряд ли будет 0
            if abs(form_dist_float) > EPSILON:
                adj_calc = (race_distance / form_dist_float) * form.calc_time
                s_adj_calcs.append(adj_calc)
                if form.index < 5: s_adj_l5_calcs.append(adj_calc)

    if s_cur_times:
        dog.cbt = min(s_cur_times); dog.cbt_source = "CUR"
    elif s_adj_calcs:
        dog.cbt = min(s_adj_calcs); dog.cbt_source = "ADJ"

    if len(s_cur_l5_times) >= 2:
        dog.bda = safe_median(s_cur_l5_times); dog.bda_mode = "STD_NO_DATE_CUR"
    elif s_cur_forms_all:
        dog.bda_mode = "ESC_CLASS_CUR"
        esc_class_times = [f.calc_time for f in s_cur_forms_all if f.race_class == race_grade and f.calc_time is not None]
        if len(esc_class_times) >= 3:
            dog.bda = safe_median(esc_class_times)
        else:
            dog.bda_mode = "ESC_DIST_CUR"; dog.bda = safe_median(s_cur_times)
    elif s_adj_l5_calcs:
        dog.bda = safe_median(s_adj_l5_calcs); dog.bda_mode = "FALLBACK_NO_DATE_ADJ"

def process_cbt_bda_scores(all_dogs: List[ProcessedDog], race_card: Dict[str, Any]):
    race_distance = int(to_float(race_card.get('distance')) or 0)
    race_grade = str(race_card.get('race_grade', 'UNKNOWN'))
    for dog in all_dogs:
        calculate_cbt_bda(dog, race_distance, race_grade)

# --- ШАГ 7: Базовый Счет (3.1, 3.2, 3.3) (v1.2) ---

def apply_improver_cap(bda: float, cbt: float) -> Tuple[float, str]:
    delta = bda - cbt; z = delta / 0.07
    bda_contribution: float = 0.0; cap_regime = "none"
    if is_le(z, 2.0):
        bda_contribution = 0.60 * z; cap_regime = "none"
    elif is_le(z, 4.0):
        bda_contribution = 0.60 * (2.0 + 0.5 * (z - 2.0)); cap_regime = "mid"
    else:
        bda_contribution = 0.60 * (3.0 + 0.25 * (z - 4.0)); cap_regime = "hard"
    if is_le(bda_contribution, -2.00):
        bda_contribution = -2.00; cap_regime = "clipped_min"
    if is_ge(bda_contribution, 3.20):
        bda_contribution = 3.20; cap_regime = "clipped_max"
    return bda_contribution, cap_regime


def calculate_early_pace(
    all_dogs: List[ProcessedDog],
    race_distance: int,
    field_size: int
) -> Tuple[str, float]:
    dogs_with_sectional_l5 = 0
    for dog in all_dogs:
        if any(
            f.valid_sectional is not None
            for f in dog.forms[:5]
            if is_eq(to_float(f.distance), race_distance)
        ):
            dogs_with_sectional_l5 += 1

    threshold = math.ceil(field_size / 2.0)
    run_mode = "Time" if (dogs_with_sectional_l5 >= threshold) else "Rank"
    global_cms_values: List[float] = []

    for dog in all_dogs:
        dog_median_values: List[float] = []
        dog_forms_l5_on_dist = [
            f for f in dog.forms[:5] if is_eq(to_float(f.distance), race_distance)
        ]
        if run_mode == "Time":
            dog_median_values = [
                f.valid_sectional for f in dog_forms_l5_on_dist if f.valid_sectional is not None
            ]
        else:
            dog_median_values = [
                float(f.bnd_pos) for f in dog_forms_l5_on_dist if f.bnd_pos is not None
            ]
        global_cms_values.extend(dog_median_values)
        dog.early_dog_median_split = safe_median(dog_median_values)

    global_cms = safe_median(global_cms_values)

    for dog in all_dogs:
        if dog.early_dog_median_split is None:
            dog_split = global_cms if global_cms is not None else 0.0
            cms = global_cms if global_cms is not None else 0.0
        else:
            dog_split = dog.early_dog_median_split
            cms = global_cms if global_cms is not None else 0.0

        if run_mode == "Time":
            early_score = (cms - dog_split) / 0.04
            dog.raw_score += 0.40 * early_score
        else:
            early_score = (cms - dog_split) / 0.5
            dog.raw_score += 0.40 * early_score

    return run_mode, (global_cms if global_cms is not None else 0.0)


def calculate_late_pace(dog: ProcessedDog):
    l3_forms = dog.forms[:3]; deltas: Dict[int, int] = {}
    for form in l3_forms:
        if form.bnd_pos is not None and form.valid_position is not None:
            deltas[form.index] = form.bnd_pos - form.valid_position
    weights = {0: 0.5, 1: 0.3, 2: 0.2}
    numerator, denominator = 0.0, 0.0
    for idx, delta in deltas.items():
        if idx in weights:
            numerator += weights[idx] * delta
            denominator += weights[idx]
    late_pace_rw: Optional[float] = None
    if denominator > EPSILON:
        late_pace_rw = numerator / denominator
    dog.late_pace_rw = late_pace_rw
    if late_pace_rw is not None:
        dog.raw_score += 0.90 * late_pace_rw
    else:
        proxy_score = 0.25 if dog.finisher_flag else 0.0
        dog.raw_score += 0.90 * proxy_score


def process_base_score(
    all_dogs: List[ProcessedDog],
    race_card: Dict[str, Any]
) -> Tuple[str, float]:
    field_size = len(all_dogs)
    for dog in all_dogs:
        dog.raw_score = 0.0
    run_mode, global_cms = calculate_early_pace(
        all_dogs,
        int(to_float(race_card.get('distance')) or 0),
        field_size
    )
    for dog in all_dogs:
        if dog.cbt is not None and dog.bda is not None:
            bda_contribution, cap_regime = apply_improver_cap(dog.bda, dog.cbt)
            dog.raw_score += bda_contribution
            dog.flags.append(f"BDA_REGIME:{cap_regime}")
        calculate_late_pace(dog)
        dog.base_score_trace = dog.raw_score
    return run_mode, global_cms


# --- ШАГ 8: Модификаторы (Разделы 4, 5) (v1.2) ---
def get_class_index_mod(grade: str) -> Optional[int]:
    if not grade or not isinstance(grade, str): return None
    grade = grade.upper()
    if grade in ("OR", "IV"): return 0
    if grade in ("HP", "T", "T1", "T2", "T3"): return None
    if grade.startswith('A') and grade[1:].isdigit(): return int(grade[1:])
    if grade.startswith('B') and grade[1:].isdigit(): return 11 + int(grade[1:])
    if grade.startswith('D') and grade[1:].isdigit(): return 17 + int(grade[1:])
    if grade.startswith('S') and grade[1:].isdigit(): return 23 + int(grade[1:])
    return None

def process_modifiers(
    all_dogs: List[ProcessedDog],
    race_card: Dict[str, Any],
    run_mode: str,
    global_cms: float
):
    field_class_medians: List[int] = []
    for dog in all_dogs:
        dog_class_indices = [get_class_index_mod(f.race_class) for f in dog.forms[:5] if get_class_index_mod(f.race_class) is not None]
        dog_median_class = safe_median(dog_class_indices)
        if dog_median_class is not None:
            field_class_medians.append(dog_median_class)
            dog.dog_median_class = dog_median_class
        else:
            dog.dog_median_class = None

    field_class_median = safe_median(field_class_medians)
    current_race_grade_idx = get_class_index_mod(race_card.get('race_grade', ''))

    for dog in all_dogs:
        mods: Dict[str, float] = {}
        l1_form = dog.forms[0] if dog.forms else None
        clean_slate = False
        if l1_form:
            l1_tokens = l1_form.tokens
            clean_slate = (not l1_tokens['trouble'] and not l1_tokens['saw'] and
                           (l1_tokens['qaw'] or l1_tokens['ep'] or l1_tokens['led']))

        front_runner_mitigation = is_ge(dog.front_rate_adj, 0.6)
        is_rails, is_wide = (dog.style == 'rails'), (dog.style == 'wide')
        is_trap_1_2, is_trap_5_6 = (dog.trap_number <= 2), (dog.trap_number >= 5)

        if (is_wide and is_trap_1_2) or (is_rails and is_trap_5_6):
            mods['4.1_draw_mismatch'] = -0.85; dog.flags.append("DRAW_MISMATCH")
        elif (is_rails and is_trap_1_2) or (is_wide and is_trap_5_6):
            mods['4.1b_draw_match'] = +0.30; dog.flags.append("DRAW_MATCH_BONUS")

        wide_count_l5 = sum(1 for f in dog.forms[:5] if f.tokens['wide'])
        dog_slower_than_cms = (run_mode == 'Time' and dog.early_dog_median_split is not None and
                               is_ge(dog.early_dog_median_split, global_cms + EPSILON))
        if wide_count_l5 >= 3 and dog_slower_than_cms:
            mods['4.1c_wide_persist_1'] = -0.25
            if (is_wide and is_trap_1_2): mods['4.1c_wide_persist_2'] = -0.10

        top3_count_l5 = sum(1 for f in dog.forms[:5] if f.valid_position and f.valid_position <= 3)
        condition_4_2 = (is_ge(dog.trouble_rate_adj, 0.4) and
                         is_le(dog.front_rate_adj, 0.4 - EPSILON) and top3_count_l5 < 2)
        if condition_4_2 and not clean_slate and not front_runner_mitigation:
            mods['4.2_trouble'] = -0.90; dog.flags.append("TROUBLE")

        cond_A_top3 = (dog.n_valid_l5 > 0) and (top3_count_l5 / dog.n_valid_l5) >= 0.5
        led_fin_cnt = sum(1 for f in dog.forms[:5] if f.tokens['led'] or f.tokens['finisher'])
        cond_A_ledfin = dog.n_valid_l5 > 0 and (led_fin_cnt / dog.n_valid_l5) >= 0.5
        if cond_A_top3 or cond_A_ledfin:
            mods['4.3_stability'] = +1.70; dog.flags.append("STABILITY_BONUS")

        if dog.dog_median_class is not None and field_class_median is not None:
            pressure = dog.dog_median_class - field_class_median
            if is_ge(pressure, 2.0): mods['5.1_class_down'] = -0.25
            elif is_le(pressure, -2.0): mods['5.1_class_up'] = +0.30

        if top3_count_l5 >= 3:
            mods['5.4_consistency'] = +0.40; dog.flags.append("CONSISTENCY")

        l3_forms = dog.forms[:3]; rwt_num, rwt_den = 0.0, 0.0
        weights = {0: 0.5, 1: 0.3, 2: 0.2}
        for form in l3_forms:
            if form.index not in weights: continue
            w = weights[form.index]; rwt_den += w
            if form.tokens['trouble'] or form.tokens['saw']: rwt_num += w
        rwt = (rwt_num / rwt_den) if rwt_den > EPSILON else 0.0

        if (is_ge(rwt, 0.55) and '4.2_trouble' not in mods and not clean_slate):
            penalty = -1.25 * (1.0 - min(dog.front_rate_adj, 0.6))
            mods['5.5_rwt'] = penalty; dog.flags.append("HIGH_TROUBLE_RATE_PENALTY")

        if dog.cbt_source == 'ADJ': mods['5.6a_data_thin_cbt'] = -1.50
        if dog.bda_mode == 'FALLBACK_NO_DATE_ADJ': mods['5.6b_data_thin_bda'] = -0.75

        pos_dyn_score, neg_dyn_score = 0.0, 0.0
        weights_l5 = {0: 1.0, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.2}
        for form in dog.forms[:5]:
            if form.index not in weights_l5: continue
            w = weights_l5[form.index]
            pos_tokens = (form.tokens['qaw'], form.tokens['ep'], form.tokens['led'], form.tokens['finisher'])
            pos_dyn_score += w * (sum(pos_tokens) * 0.40)
            if form.tokens.get('trouble_soft'):  neg_dyn_score -= w * 0.05
            if form.tokens.get('trouble_heavy'): neg_dyn_score -= w * 0.10

        mods['5.7_comment_pos'] = min(pos_dyn_score, 1.60)
        mods['5.7_comment_neg'] = neg_dyn_score

        forms_l2 = [f for f in dog.forms[:2] if f.valid_position is not None]
        if len(forms_l2) == 2:
            if (forms_l2[0].valid_position == 1 or forms_l2[1].valid_position == 1):
                mods['5.8_momentum'] = +1.20; dog.flags.append("MOMENTUM_BONUS")
            elif (forms_l2[0].valid_position <= 3 and forms_l2[1].valid_position <= 3 and
                  forms_l2[0].valid_position < forms_l2[1].valid_position):
                mods['5.8_momentum'] = +1.20; dog.flags.append("MOMENTUM_BONUS")

        if (dog.dog_median_class is not None and current_race_grade_idx is not None):
            if (dog.dog_median_class - current_race_grade_idx) >= 3:
                 mods['5.9_step_up'] = -1.00

        race_dist = int(to_float(race_card.get('distance')) or 0)
        if run_mode == "Time":
            times5 = [f.valid_sectional for f in dog.forms[:5] if is_eq(to_float(f.distance), race_dist) and f.valid_sectional]
            if times5:
                min5 = min(times5)
                times3 = [f.valid_sectional for f in dog.forms[:3] if is_eq(to_float(f.distance), race_dist) and f.valid_sectional]
                if times3 and is_eq(min5, min(times3)) and is_le(min5, global_cms - 0.06):
                    mods['5.12_ssb'] = +0.60; dog.flags.append("SSB")
        else:
            if any(f.bnd_pos == 1 for f in dog.forms[:3] if is_eq(to_float(f.distance), race_dist)):
                 mods['5.12_ssb'] = +0.45; dog.flags.append("SSB")

        if l1_form and (l1_form.tokens['saw'] or l1_form.tokens['trouble']) and (l1_form.tokens['led']):
            mods['5.13_erb'] = +0.25

        dog.mods_trace = mods

        stability_flag = '4.3_stability' in mods
        consistency_flag = '5.4_consistency' in mods

        if (dog.late_pace_rw is not None and stability_flag and consistency_flag):
            original_rw = dog.late_pace_rw
            shielded_rw = max(original_rw, -1.10)
            if not is_eq(original_rw, shielded_rw):
                correction = 0.90 * (shielded_rw - original_rw)
                dog.base_score_trace += correction
                dog.late_pace_rw = shielded_rw

# --- ШАГ 9: Валидация и "Карантин" (Раздел 8) ---
def run_validation_protocol(all_dogs: List[ProcessedDog], race_distance: int) -> Tuple[List[ProcessedDog], List[ProcessedDog]]:
    ranked_dogs, quarantined_dogs = [], []
    dist_min, dist_max = race_distance * 0.8, race_distance * 1.2
    for dog in all_dogs:
        has_competitive_form, has_any_runtime = False, False
        adj_competitive_forms_count = 0
        for form in dog.forms:
            if form.calc_time is not None: has_any_runtime = True
            is_comp = form.race_class not in ("HP", "T", "T1", "T2", "T3")
            if is_comp: has_competitive_form = True
            form_dist_float = to_float(form.distance)
            if form_dist_float is None: continue
            is_adj = (is_ge(form_dist_float, dist_min) and is_le(form_dist_float, dist_max) and not is_eq(form_dist_float, race_distance))
            if is_comp and is_adj and form.calc_time is not None:
                 adj_competitive_forms_count += 1
        is_quarantined = (not has_competitive_form) and (not has_any_runtime)
        if is_quarantined and adj_competitive_forms_count >= 3:
            is_quarantined = False
            dog.flags.append("DATA_THIN_RANKED"); dog.flags.append("DISTANCE_FALLBACK_USED")
        cond_8_5_a = is_ge(dog.front_rate_adj, 0.6)
        cond_8_5_b = "SSB" in dog.flags
        if (is_quarantined and (not has_competitive_form) and has_any_runtime and
            (cond_8_5_a or cond_8_5_b)):
            is_quarantined = False
            dog.flags.append("NON_COMPETITIVE_DATA_ONLY"); dog.flags.append("HP_T_SOFT_INCLUDED")
            penalty = -1.00
            dog.mods_trace['8.5_penalty'] = penalty
        if is_quarantined:
            dog.flags.append("UNRANKED_INSUFFICIENT_DATA"); quarantined_dogs.append(dog)
        else:
            ranked_dogs.append(dog)
    return ranked_dogs, quarantined_dogs

# --- ШАГ 10: Финальные Капы (5.11, 6.1) (ИСПРАВЛЕНО v1.3) ---
def apply_final_caps(ranked_dogs: List[ProcessedDog]):
    """
    (Патч v1.3: Баг - потеря 5.x < 0)
    """
    TROUBLE_CAP_KEYS = ['4.2_trouble', '5.5_rwt', '5.7_comment_neg', '8.5_penalty']

    for dog in ranked_dogs:
        base_score = dog.base_score_trace
        trouble_cap_sum = 0.0; bonus_cap_sum = 0.0; other_mods_sum = 0.0

        for key, value in dog.mods_trace.items():
            if key in TROUBLE_CAP_KEYS:
                trouble_cap_sum += value
            elif key.startswith('5.'):
                if value > EPSILON:
                    bonus_cap_sum += value
                else: # <-- (Патч v1.3: Учитываем 5.x < 0)
                    other_mods_sum += value
            elif key.startswith('4.'):
                other_mods_sum += value
            # (Игнорируем 3.x)

        # --- 5.11: Trouble Cap ---
        if is_le(trouble_cap_sum, -2.5):
            trouble_cap_total = -2.5; dog.flags.append("TROUBLE_CAP_APPLIED")
        else:
            trouble_cap_total = trouble_cap_sum
        dog.trouble_cap_total_trace = trouble_cap_total

        # --- 6.1: Bonus Cap ---
        if is_ge(bonus_cap_sum, 4.0 + EPSILON):
            bonus_cap_total = 4.0; dog.flags.append("BONUS_CAP_APPLIED")
        else:
            bonus_cap_total = bonus_cap_sum
        dog.bonus_cap_total_trace = bonus_cap_total

        # --- Финальное суммирование ---
        dog.raw_score = (
            base_score +
            other_mods_sum +
            trouble_cap_total +
            bonus_cap_total
        )

# --- ШАГ 11: Softmax и Ранжирование (D10, 6.2, 6.3) ---
def calculate_softmax_and_rank(ranked_dogs: List[ProcessedDog]) -> List[ProcessedDog]:
    if not ranked_dogs: return []
    for dog in ranked_dogs:
        dog.neg_flags_count = sum(1 for f in dog.flags if f in ("TROUBLE", "NEG_START", "DRAW_MISMATCH"))
        dog.pos_flags_count = sum(1 for f in dog.flags if f in ("STABILITY_BONUS"))

    scores = [dog.raw_score for dog in ranked_dogs]
    median_raw = safe_median(scores) or 0.0
    exp_sum = 0.0; pre_softmax = {}
    for dog in ranked_dogs:
        # Защита от слишком больших/малых raw_score, которые ломают exp()
        score = max(-700.0, min(700.0, dog.raw_score))
        exp_val = math.exp(score)
        pre_softmax[dog.dog_name] = {'exp': exp_val, 'raw': dog.raw_score}
        exp_sum += exp_val

    if exp_sum < EPSILON:
        p = 100.0 / len(ranked_dogs)
        for dog in ranked_dogs: dog.percentage = p
    else:
        raw_percentages = {}
        for dog in ranked_dogs:
            name = dog.dog_name; data = pre_softmax[name]
            p_i = (data['exp'] / exp_sum) * 100.0
            if is_ge(data['raw'], median_raw + 1.0 + EPSILON): p_i *= 1.1
            elif is_le(data['raw'], median_raw - 0.8 - EPSILON): p_i *= 0.8
            raw_percentages[name] = p_i

        p_sum = sum(raw_percentages.values())
        if p_sum < EPSILON: p_sum = 1.0
        total_points_to_distribute = 1000; distributed_points = 0
        remainders = {}

        for dog in ranked_dogs:
            name = dog.dog_name
            norm_p = (raw_percentages[name] / p_sum) * 100.0
            points_float = norm_p * 10.0
            points_int = math.floor(points_float)
            dog.percentage = points_int / 10.0
            remainders[name] = points_float - points_int
            distributed_points += points_int

        points_to_add = total_points_to_distribute - distributed_points
        sorted_by_remainder = sorted(
            ranked_dogs,
            key=lambda d: (-remainders[d.dog_name], -d.raw_score, d.neg_flags_count, -d.pos_flags_count, d.dog_name)
        )
        for i in range(int(round(points_to_add))):
            if i < len(sorted_by_remainder):
                sorted_by_remainder[i].percentage += 0.1

    final_ranked_list = sorted(
        ranked_dogs,
        key=lambda d: (-d.percentage, -d.raw_score, d.neg_flags_count, -d.pos_flags_count, d.dog_name)
    )
    for i, dog in enumerate(final_ranked_list):
        dog.rank = i + 1
        dog.percentage = round(dog.percentage, 1)
    return final_ranked_list

# --- (Патч: D12 - Хеш трассировки) ---
def make_trace_hash(trace: dict) -> str:
    # Округляем все float до 4 знаков для стабильности хеша
    stable_trace = {}
    for k, v in trace.items():
        if isinstance(v, float):
            stable_trace[k] = round(v, 4)
        elif isinstance(v, dict):
            stable_trace[k] = {dk: round(dv, 4) if isinstance(dv, float) else dv for dk, dv in v.items()}
        else:
            stable_trace[k] = v

    blob = json.dumps(stable_trace, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()

# --- (Патч: п.0 - "Железный занавес") ---
def pre_computation_check(all_dogs: List[ProcessedDog], race_distance: int) -> Tuple[bool, List[Dict[str, Any]]]:
    dist_min, dist_max = race_distance * 0.8, race_distance * 1.2
    results = []
    stop = False
    for d in all_dogs:
        ok = False
        for f in d.forms:
            form_dist_float = to_float(f.distance)
            if f.calc_time is not None and form_dist_float is not None:
                 if is_eq(form_dist_float, race_distance) or (is_ge(form_dist_float, dist_min) and is_le(form_dist_float, dist_max)):
                     ok = True
                     break # Нашли хотя бы одну, достаточно
        results.append({"dogName": d.dog_name, "rule_path_exists": ok})
        if not ok:
            stop = True
    return stop, results

# --- ШАГ 12: Финальная Сборка (Главная функция) (v1.2) ---
def generate_predictions(race_card: Dict[str, Any]) -> Dict[str, Any]:
    if not race_card or 'dogs' not in race_card or 'distance' not in race_card:
        return {"predictions": [], "trace": {"error": "Invalid Card format"}}

    race_dist_float = to_float(race_card.get('distance'))
    if race_dist_float is None:
        return {"predictions": [], "trace": {"error": "Invalid race distance"}}
    race_distance = int(race_dist_float)

    all_dogs = process_card(race_card)

    stop, pre_check_results = pre_computation_check(all_dogs, race_distance)
    if stop:
        failed_dog = next((r['dogName'] for r in pre_check_results if not r['rule_path_exists']), "UNKNOWN")
        return {
            "error": f"ОСТАНОВКА. Нет пути расчета для собаки: {failed_dog}",
            "pre_computation_check": pre_check_results,
            "predictions": []
        }

    for dog in all_dogs:
        dog.flags.extend(["NO_DATE_FIELDS", "ORDER_BY_INDEX_USED"])

    process_aggregates(all_dogs)
    process_cbt_bda_scores(all_dogs, race_card)
    run_mode, global_cms = process_base_score(all_dogs, race_card)
    process_modifiers(all_dogs, race_card, run_mode, global_cms)
    ranked_dogs, quarantined_dogs = run_validation_protocol(all_dogs, race_distance)
    apply_final_caps(ranked_dogs)
    final_ranked_list = calculate_softmax_and_rank(ranked_dogs)

    predictions = []
    for dog in final_ranked_list:
        final_flags = sorted([f for f in dog.flags if not f.startswith('BDA_REGIME:')])
        predictions.append({
            "dogName": dog.dog_name,
            "raw_score": round(dog.raw_score, 4),
            "percentage": dog.percentage,
            "rank": dog.rank,
            "flags": final_flags
        })
    for dog in quarantined_dogs:
        predictions.append({
            "dogName": dog.dog_name, "raw_score": None, "percentage": None, "rank": None,
            "flags": sorted([f for f in dog.flags if "UNRANKED" in f or "NO_DATE" in f])
        })

    trace = {
        "run_mode": run_mode,
        "global_cms": global_cms,
        "pre_check": pre_check_results,
        "raw_scores": {d.dog_name: round(d.raw_score, 4) for d in ranked_dogs},
        "base_scores_trace": {d.dog_name: round(d.base_score_trace, 4) for d in ranked_dogs},
        "bda_modes": {d.dog_name: d.bda_mode for d in all_dogs},
        "ssb_active": {d.dog_name: "SSB" in d.flags for d in all_dogs},
        "neg_start_L1": {d.dog_name: d.neg_start_flag for d in all_dogs},
        "styles": {d.dog_name: d.style for d in all_dogs},
        "trouble_cap_totals": {d.dog_name: round(d.trouble_cap_total_trace, 4) for d in ranked_dogs},
        "bonus_cap_totals": {d.dog_name: round(d.bonus_cap_total_trace, 4) for d in ranked_dogs},
    }

    trace["trace_hash"] = make_trace_hash(trace)
    return {"predictions": predictions, "trace": trace}
