import os
import time

import vertexai
import vertexai.generative_models as gai
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature

_API_KEY=os.getenv("LTC_GOOGLE_API_KEY")


class Gemini:
    def __init__(
        self, text: str, prompt: str, hlf_cfg: dict[str, HandcraftedLinguisticFeature]
    ) -> None:
        self._input_text = text
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")
        vertexai.init(project="llm-tst-consistency", location="us-central1")
        self._model = gai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=[prompt]
        )

    @staticmethod
    def _apply_prefix(name: str, prefix: str) -> str:
        if prefix is None or len(prefix.strip()) < 1:
            return name
        return f"{prefix}_{name}"

    def __call__(self, prefix: str = "") -> dict[str, float]:
        time.sleep(12)
        response = self._model.generate_content(self._input_text)
        doc = self._nlp(response.text)
        return {
            self._apply_prefix(name, prefix): hlf(doc)
            for name, hlf in self._hlf_cfg.items()
        }