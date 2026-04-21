"""
Модуль семантического сходства — на базе Sentence Transformers.

Сравнивает предложение с эталонными прототипами значений,
чтобы определить, употреблено ли «лапша» в переносном (обман)
или прямом (еда) значении.

Модель скачивается один раз и кэшируется локально в model_cache/.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, MODEL_CACHE_DIR, IDIOM_MEANINGS


def load_semantic_model() -> SentenceTransformer:
    """Загружает модель из локального кэша; скачивает только при первом запуске."""
    return SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)


def _mean_embedding(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Возвращает усреднённый эмбеддинг списка текстов."""
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.mean(axis=0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство между двумя одномерными векторами."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class MeaningClassifier:
    """Предвычисляет эталонные эмбеддинги один раз, затем быстро классифицирует."""

    def __init__(self, model: SentenceTransformer, meanings: dict[str, list[str]]):
        self.model = model
        # Предвычисляем эталонные векторы один раз
        self._fig_emb = _mean_embedding(model, meanings["figurative"])
        self._lit_emb = _mean_embedding(model, meanings["literal"])

    def classify(self, sentence: str) -> str:
        """Возвращает ``"figurative"`` или ``"literal"``."""
        sent_emb = self.model.encode(sentence, convert_to_numpy=True)
        sim_fig = _cosine_similarity(sent_emb, self._fig_emb)
        sim_lit = _cosine_similarity(sent_emb, self._lit_emb)
        return "figurative" if sim_fig > sim_lit else "literal"


def classify_by_meaning(
    sentence: str,
    model: SentenceTransformer,
    meanings: dict[str, list[str]] | None = None,
) -> str:
    """Возвращает ``"figurative"`` или ``"literal"`` в зависимости от
    семантической близости предложения к эталонным фразам из *meanings*.
    """
    if meanings is None:
        meanings = IDIOM_MEANINGS

    sent_emb = model.encode(sentence, convert_to_numpy=True)

    fig_emb = _mean_embedding(model, meanings["figurative"])
    lit_emb = _mean_embedding(model, meanings["literal"])

    sim_fig = _cosine_similarity(sent_emb, fig_emb)
    sim_lit = _cosine_similarity(sent_emb, lit_emb)

    return "figurative" if sim_fig > sim_lit else "literal"
