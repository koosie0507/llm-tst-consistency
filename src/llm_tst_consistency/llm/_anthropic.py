import time

import anthropic

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel, DEFAULT_NLP_MODEL


class Claude3(BaseModel):
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
            nlp_model_name=kwargs.get("nlp_model_name")
        )
        self._system_prompt = prompt
        self._messages = [{"role": "user", "content": text}]
        self._client = anthropic.Client(api_key=api_key)

    def _generate_text(self) -> str:
        time.sleep(5)
        response = self._client.messages.create(
            model=self._model_name,
            system=self._system_prompt,
            messages=self._messages,
            max_tokens=1024,
        )
        return response.content[0].text
