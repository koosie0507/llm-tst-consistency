import ollama
from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel


class Ollama(BaseModel):
    def __init__(
        self,
        host: str,
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
        self._messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        self._client = ollama.Client(host=host)

    def _generate_text(self) -> str:
        response = self._client.chat(model=self._model_name, messages=self._messages)
        return response["message"]["content"]
