import os
import time

import anthropic
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature


class Claude3:
    def __init__(
        self, text: str, prompt: str, hlf_cfg: dict[str, HandcraftedLinguisticFeature]
    ) -> None:
        self._system_prompt = prompt
        self._messages = [{"role": "user", "content": text}]
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")
        self._client = anthropic.Client(api_key=os.getenv("LTC_ANTHROPIC_KEY"))

    @staticmethod
    def _apply_prefix(name: str, prefix: str) -> str:
        if prefix is None or len(prefix.strip()) < 1:
            return name
        return f"{prefix}_{name}"

    def __call__(self, prefix: str = "") -> dict[str, float]:
        time.sleep(5)
        response = self._client.messages.create(
            model="claude-3-opus-20240229",
            system=self._system_prompt,
            messages=self._messages,
            max_tokens=1024,
        )
        doc = self._nlp(response.content[0].text)
        return {
            self._apply_prefix(name, prefix): hlf(doc)
            for name, hlf in self._hlf_cfg.items()
        }