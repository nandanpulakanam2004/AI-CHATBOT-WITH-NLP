from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

from utils import ensure_nltk_data, preprocess_text


LOW_CONFIDENCE_MESSAGE = "Sorry, I'm not sure I understood that. Can you rephrase?"


@dataclass(frozen=True)
class Intent:
    tag: str
    patterns: List[str]
    responses: List[str]


@dataclass(frozen=True)
class MatchDetails:
    intent: str
    confidence: float  # 0..1 (classifier probability)
    similarity: float  # 0..1
    effective_confidence: float  # 0..1 (used for fallback decision)
    method: str  # "classifier"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_intents(path: Path) -> List[Intent]:
    data = json.loads(path.read_text(encoding="utf-8"))
    intents: List[Intent] = []
    for item in data.get("intents", []):
        intents.append(
            Intent(
                tag=item["tag"],
                patterns=list(item.get("patterns", [])),
                responses=list(item.get("responses", [])),
            )
        )
    if not intents:
        raise ValueError("No intents found. Check intents.json format.")
    return intents


def _build_training_data(intents: List[Intent]) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    for intent in intents:
        for p in intent.patterns:
            texts.append(preprocess_text(p))
            labels.append(intent.tag)
    return texts, labels


def _placeholders() -> Dict[str, str]:
    now = datetime.now()
    return {
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "date": now.strftime("%A, %d %B %Y"),
    }


def _render_response(template: str, user_input: str) -> str:
    values = _placeholders()
    values["user_input"] = user_input.strip()
    try:
        return template.format(**values)
    except Exception:
        return template


class ChatbotModel:
    """
    Intent chatbot with:
    - preprocessing (utils.py)
    - TF-IDF features
    - Logistic Regression classifier (predict_proba => confidence)
    - cosine similarity to closest pattern (debug/extra signal)
    - model persistence (joblib) so it doesn't retrain every run
    """

    def __init__(
        self,
        *,
        intents_path: Path,
        model_dir: Optional[Path] = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        ensure_nltk_data()

        self.intents_path = intents_path
        self.model_dir = model_dir or intents_path.parent / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.confidence_threshold = float(confidence_threshold)
        self.intents = load_intents(intents_path)
        self.intent_by_tag = {i.tag: i for i in self.intents}

        self._model_path = self.model_dir / "intent_model.joblib"
        self._meta_path = self.model_dir / "intent_model_meta.json"

        self.vectorizer: TfidfVectorizer
        self.classifier: LogisticRegression
        self.pattern_matrix = None  # sparse matrix
        self.labels = None  # np.ndarray[str]
        self.pattern_texts: List[str] = []

        self._load_or_train()

    def _load_or_train(self) -> None:
        intents_hash = _hash_file(self.intents_path)

        if self._model_path.exists() and self._meta_path.exists():
            try:
                meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                if meta.get("intents_hash") == intents_hash:
                    payload = load(self._model_path)
                    self.vectorizer = payload["vectorizer"]
                    self.classifier = payload["classifier"]
                    self.pattern_matrix = payload["pattern_matrix"]
                    self.labels = payload["labels"]
                    self.pattern_texts = payload.get("pattern_texts", [])
                    return
            except Exception:
                # If anything goes wrong, retrain cleanly.
                pass

        self._train_and_save(intents_hash=intents_hash)

    def _train_and_save(self, *, intents_hash: str) -> None:
        train_texts, train_labels = _build_training_data(self.intents)

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.pattern_matrix = self.vectorizer.fit_transform(train_texts)

        self.labels = np.array(train_labels)
        self.pattern_texts = list(train_texts)

        self.classifier = LogisticRegression(max_iter=3000)
        self.classifier.fit(self.pattern_matrix, self.labels)

        dump(
            {
                "vectorizer": self.vectorizer,
                "classifier": self.classifier,
                "pattern_matrix": self.pattern_matrix,
                "labels": self.labels,
                "pattern_texts": self.pattern_texts,
            },
            self._model_path,
        )
        self._meta_path.write_text(
            json.dumps(
                {
                    "intents_hash": intents_hash,
                    "trained_at": datetime.now().isoformat(timespec="seconds"),
                    "num_patterns": int(self.pattern_matrix.shape[0]),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def predict(self, user_text: str) -> MatchDetails:
        """
        Predict intent with:
        - confidence: max predict_proba
        - similarity: cosine to closest training pattern
        """
        try:
            vec = self.vectorizer.transform([preprocess_text(user_text)])
            probs = self.classifier.predict_proba(vec)[0]
            best_idx = int(np.argmax(probs))
            intent = str(self.classifier.classes_[best_idx])
            confidence = float(probs[best_idx])

            sims = cosine_similarity(vec, self.pattern_matrix)[0]
            similarity = float(np.max(sims)) if sims.size else 0.0
            effective_confidence = float(max(confidence, similarity))

            return MatchDetails(
                intent=intent,
                confidence=confidence,
                similarity=similarity,
                effective_confidence=effective_confidence,
                method="classifier",
            )
        except Exception:
            # Robust fallback: never crash.
            return MatchDetails(
                intent="fallback",
                confidence=0.0,
                similarity=0.0,
                effective_confidence=0.0,
                method="fallback",
            )

    def get_response(self, user_text: str) -> Tuple[str, MatchDetails]:
        """
        Returns (response, details). Enforces low-confidence fallback behavior.
        """
        details = self.predict(user_text)

        if details.effective_confidence < self.confidence_threshold:
            return LOW_CONFIDENCE_MESSAGE, details

        intent = self.intent_by_tag.get(details.intent) or self.intent_by_tag.get("fallback")
        if not intent or not intent.responses:
            return LOW_CONFIDENCE_MESSAGE, details

        template = random.choice(intent.responses)
        return _render_response(template, user_text), details

