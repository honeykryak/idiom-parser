"""
Модуль морфологического анализа — на базе Natasha.

Обеспечивает лемматизацию и позиционный анализ токенов для
обнаружения полной идиомы «вешать лапшу (на уши)».
"""

from dataclasses import dataclass

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc,
)

from config import HANG_VERB_LEMMAS, HANG_VERB_STEM, FOOD_STEMS


@dataclass
class TokenInfo:
    """Информация о токене: позиция, текст, лемма."""
    index: int
    text: str
    lemma: str


@dataclass
class MorphPipeline:
    """Набор переиспользуемых компонентов Natasha."""
    segmenter: Segmenter
    morph_vocab: MorphVocab
    emb: NewsEmbedding
    morph_tagger: NewsMorphTagger


def build_morph_pipeline() -> MorphPipeline:
    """Инициализирует компоненты Natasha (вызывается один раз)."""
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    return MorphPipeline(segmenter, morph_vocab, emb, morph_tagger)


def analyze_tokens(text: str, pipeline: MorphPipeline) -> list[TokenInfo]:
    """Возвращает список TokenInfo с позицией, текстом и леммой каждого токена."""
    doc = Doc(text)
    doc.segment(pipeline.segmenter)
    doc.tag_morph(pipeline.morph_tagger)
    for token in doc.tokens:
        token.lemmatize(pipeline.morph_vocab)
    return [
        TokenInfo(i, token.text.lower(), (token.lemma or token.text).lower())
        for i, token in enumerate(doc.tokens)
    ]


def _is_hang_verb(token: TokenInfo) -> bool:
    """Проверяет, является ли токен формой глагола «вешать» (двухслойная проверка)."""
    # Слой 1: точное совпадение леммы
    if token.lemma in HANG_VERB_LEMMAS:
        return True
    # Слой 2: основа «веш» в сырой форме (ловит случаи, когда Natasha не лемматизировала)
    if HANG_VERB_STEM in token.text:
        return True
    return False


def find_lapsha_positions(tokens: list[TokenInfo]) -> list[int]:
    """Возвращает индексы токенов с леммой «лапша»."""
    return [t.index for t in tokens if t.lemma == "лапша"]


def find_hang_verb_positions(tokens: list[TokenInfo]) -> list[int]:
    """Возвращает индексы токенов-глаголов «вешать» и однокоренных."""
    return [t.index for t in tokens if _is_hang_verb(t)]


def find_uho_positions(tokens: list[TokenInfo]) -> list[int]:
    """Возвращает индексы токенов с леммой «ухо»."""
    return [t.index for t in tokens if t.lemma == "ухо"]


def find_food_positions(tokens: list[TokenInfo]) -> list[int]:
    """Возвращает индексы токенов, содержащих пищевые маркеры."""
    result = []
    for t in tokens:
        for stem in FOOD_STEMS:
            if stem in t.text or stem in t.lemma:
                result.append(t.index)
                break
    return result


def min_distance(positions_a: list[int], positions_b: list[int]) -> int | None:
    """Минимальное расстояние (в токенах) между двумя группами позиций.

    Возвращает None, если одна из групп пуста.
    """
    if not positions_a or not positions_b:
        return None
    return min(abs(a - b) for a in positions_a for b in positions_b)
