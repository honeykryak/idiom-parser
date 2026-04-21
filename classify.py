"""
Детектор фразеологических выражений — «вешать лапшу на уши»

Классифицирует предложения, содержащие слово «лапша», на три группы:
  1. Идиома:       полная идиома («лапша» + глагол «вешать/повесить»)
  2. Сегментация:  «лапша» в переносном значении (обман), без глагола «вешать»
  3. Прямое:       «лапша» в прямом значении (еда, продукт)

Запуск:
    python classify.py <входной_файл.xlsx>

Результат:
    idiom.xlsx, segmentation.xlsx, literal.xlsx — сохраняются рядом с входным файлом.
"""

import sys
import os

from config import IDIOM_MEANINGS, DEFAULT_MAX_DISTANCE, UHO_MAX_DISTANCE, FOOD_MAX_DISTANCE
from morphology import (
    build_morph_pipeline, analyze_tokens,
    find_lapsha_positions, find_hang_verb_positions,
    find_uho_positions, find_food_positions, min_distance,
)
from semantics import load_semantic_model, MeaningClassifier
from io_excel import read_sentences, write_sentences


# ---------------------------------------------------------------------------
# Классификация одного предложения по морфологическим признакам
# ---------------------------------------------------------------------------

def classify_morphology(tokens, max_dist: int) -> str:
    """Возвращает 'idiom', 'segmentation', 'literal' или 'unknown'.

    Порядок проверки:
      1. лапша + глагол + ухо (любое расстояние)  → idiom
      2. лапша + глагол (≤ max_dist)             → idiom
      3. лапша + ухо (≤ UHO_MAX_DISTANCE, нет гл.) → segmentation
      4. лапша + пищевой маркер (≤ FOOD_MAX_DISTANCE) → literal
      5. иначе                                   → unknown (для семантической модели)
    """
    lapsha_pos = find_lapsha_positions(tokens)
    if not lapsha_pos:
        return "unknown"

    verb_pos = find_hang_verb_positions(tokens)
    uho_pos = find_uho_positions(tokens)
    food_pos = find_food_positions(tokens)

    has_verb = bool(verb_pos)
    has_uho = bool(uho_pos)

    # Правило 1: все три компонента — однозначно идиома (любое расстояние)
    if has_verb and has_uho:
        return "idiom"

    # Правило 2: лапша + глагол близко
    if has_verb:
        dist = min_distance(lapsha_pos, verb_pos)
        if dist is not None and dist <= max_dist:
            return "idiom"

    # Правило 3: лапша + ухо близко, без глагола
    if has_uho and not has_verb:
        dist = min_distance(lapsha_pos, uho_pos)
        if dist is not None and dist <= UHO_MAX_DISTANCE:
            return "segmentation"

    # Правило 4: лапша + пищевой маркер близко
    dist = min_distance(lapsha_pos, food_pos)
    if dist is not None and dist <= FOOD_MAX_DISTANCE:
        return "literal"

    return "unknown"


# ---------------------------------------------------------------------------
# Семантический фильтр: сегментация vs прямое значение
# ---------------------------------------------------------------------------

def split_remaining(sentences: list[str], classifier: MeaningClassifier) -> tuple[list[str], list[str]]:
    """Разделяет оставшиеся предложения на группы сегментации и прямого значения."""
    segmentation, literal = [], []
    for sent in sentences:
        group = classifier.classify(sent)
        if group == "figurative":
            segmentation.append(sent)
        else:
            literal.append(sent)
    return segmentation, literal


# ---------------------------------------------------------------------------
# Разбор аргументов
# ---------------------------------------------------------------------------

def parse_args() -> tuple[str, int]:
    """Возвращает (путь_к_файлу, макс_расстояние)."""
    if len(sys.argv) < 2:
        print("Использование: python classify.py <входной_файл.xlsx> [макс_расстояние]")
        print(f"  макс_расстояние — максимум слов между «лапша» и «вешать» (по умолчанию {DEFAULT_MAX_DISTANCE})")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Файл не найден: {input_path}")
        sys.exit(1)

    max_dist = DEFAULT_MAX_DISTANCE
    if len(sys.argv) >= 3:
        try:
            max_dist = int(sys.argv[2])
        except ValueError:
            print(f"Неверное значение макс_расстояния: {sys.argv[2]}")
            sys.exit(1)

    return input_path, max_dist


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    input_path, max_dist = parse_args()
    output_dir = os.path.dirname(os.path.abspath(input_path))

    # --- чтение ---
    print("[1/5] Чтение входного файла …")
    sentences = read_sentences(input_path)
    print(f"      Загружено предложений: {len(sentences)}")

    # --- морфология ---
    print("[2/5] Инициализация морфологического пайплайна Natasha …")
    morph_pipeline = build_morph_pipeline()

    print(f"[3/5] Классификация по морфологии (макс. расстояние = {max_dist}) …")
    idioms, segmentation_morph, literal_morph, remaining = [], [], [], []
    for sent in sentences:
        tokens = analyze_tokens(sent, morph_pipeline)
        group = classify_morphology(tokens, max_dist)
        if group == "idiom":
            idioms.append(sent)
        elif group == "segmentation":
            segmentation_morph.append(sent)
        elif group == "literal":
            literal_morph.append(sent)
        else:
            remaining.append(sent)
    print(f"      Идиомы: {len(idioms)}, сегментация (ухо): {len(segmentation_morph)}, "
          f"прямое (еда): {len(literal_morph)}, осталось: {len(remaining)}")

    # --- семантика ---
    print("[4/5] Загрузка семантической модели и классификация оставшихся …")
    sem_model = load_semantic_model()
    classifier = MeaningClassifier(sem_model, IDIOM_MEANINGS)
    segmentation_sem, literal_sem = split_remaining(remaining, classifier)
    segmentation = segmentation_morph + segmentation_sem
    literal = literal_morph + literal_sem
    print(f"      Сегментация (сем.): {len(segmentation_sem)}, Прямое (сем.): {len(literal_sem)}")
    print(f"      Итого: идиомы={len(idioms)}, сегментация={len(segmentation)}, прямое={len(literal)}")

    # --- запись ---
    print("[5/5] Запись выходных файлов …")
    if not os.path.exists(os.path.join(output_dir, "output")):
        os.mkdir(os.path.join(output_dir, "output"))
    write_sentences(idioms,       os.path.join(output_dir, "output/idiom.xlsx"))
    write_sentences(segmentation, os.path.join(output_dir, "output/segmentation.xlsx"))
    write_sentences(literal,      os.path.join(output_dir, "output/literal.xlsx"))
    print("Готово.")


if __name__ == "__main__":
    main()
