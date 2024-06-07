import os

import openai
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature


class OpenAI:
    def __init__(
        self, text: str, prompt: str, hlf_cfg: dict[str, HandcraftedLinguisticFeature]
    ) -> None:
        self._messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")
        self._client = openai.Client(api_key=os.getenv("LTC_OPENAI_KEY"))

    @staticmethod
    def _apply_prefix(name: str, prefix: str) -> str:
        if prefix is None or len(prefix.strip()) < 1:
            return name
        return f"{prefix}_{name}"

    def __call__(self, prefix: str = "") -> dict[str, float]:
        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo", messages=self._messages
        )
        doc = self._nlp(response.choices[0].message.content)
        return {
            self._apply_prefix(name, prefix): hlf(doc)
            for name, hlf in self._hlf_cfg.items()
        }
