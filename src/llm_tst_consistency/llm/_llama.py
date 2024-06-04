import ollama
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature


class Ollama:
    def __init__(
        self, text: str, prompt: str, hlf_cfg: dict[str, HandcraftedLinguisticFeature]
    ) -> None:
        self._messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")

    def __call__(self) -> dict[str, float]:
        response = ollama.chat(model="llama3", messages=self._messages)
        doc = self._nlp(response["message"]["content"])
        return {
            name: hlf(doc)
            for name, hlf in self._hlf_cfg.items()
        }
