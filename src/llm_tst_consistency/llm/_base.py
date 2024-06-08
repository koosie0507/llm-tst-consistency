from abc import abstractmethod

import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature


DEFAULT_NLP_MODEL = "en_core_web_sm"


class BaseModel:
    def __init__(
        self,
        model_name: str,
        hlf_cfg: dict[str, HandcraftedLinguisticFeature],
        metric_level: str = None,
        nlp_model_name: str = None,
    ) -> None:
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load(nlp_model_name or DEFAULT_NLP_MODEL)
        self._model_name = model_name
        self._prefix = metric_level

    def _apply_prefix(self, name: str) -> str:
        if self._prefix is None or len(self._prefix.strip()) < 1:
            return name
        return f"{self._prefix}_{name}"

    @abstractmethod
    def _generate_text(self) -> str:
        pass

    def __call__(self) -> dict[str, float]:
        doc = self._nlp(self._generate_text())
        return {
            self._apply_prefix(name): hlf(doc) for name, hlf in self._hlf_cfg.items()
        }
