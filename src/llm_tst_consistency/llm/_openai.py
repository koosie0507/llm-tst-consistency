import openai
import spacy

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel


class GPT(BaseModel):
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
        self._messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        self._hlf_cfg = hlf_cfg
        self._nlp = spacy.load("en_core_web_sm")
        self._client = openai.Client(api_key=api_key)

    def _generate_text(self) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name, messages=self._messages
        )
        return response.choices[0].message.content
