import time

import cohere
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel


class CommandR(BaseModel):
    def __init__(
        self,
        api_key: str,
        model_name: str,
        text: str,
        prompt: str,
        hlf_cfg: dict[str, HandcraftedLinguisticFeature],
        **kwargs: str
    ) -> None:
        super().__init__(
            model_name,
            hlf_cfg,
            metric_level=kwargs.get("metric_level"),
            nlp_model_name=kwargs.get("nlp_model_name"),
        )
        self._message = text
        self._preamble = prompt
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")
        self._client = cohere.Client(api_key=api_key)

    def _generate_text(self) -> str:
        time.sleep(12)
        response = self._client.chat(
            model=self._model_name, message=self._message, preamble=self._preamble
        )
        return response.text
