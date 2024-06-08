import time

import vertexai
import vertexai.generative_models as gai

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel, DEFAULT_NLP_MODEL


class Gemini(BaseModel):
    def __init__(
        self,
        project_name: str,
        location: str,
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
        self._input_text = text
        vertexai.init(project=project_name, location=location)
        self._model = gai.GenerativeModel(
            model_name=self._model_name, system_instruction=[prompt]
        )

    def _generate_text(self) -> str:
        time.sleep(12)
        response = self._model.generate_content(self._input_text)
        return response.text
