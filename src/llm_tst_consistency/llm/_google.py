import time

import vertexai
import vertexai.generative_models as gai

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm._base import BaseModel


safety_config = [
    gai.SafetySetting(
        category=gai.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=gai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gai.SafetySetting(
        category=gai.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=gai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gai.SafetySetting(
        category=gai.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=gai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gai.SafetySetting(
        category=gai.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=gai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    gai.SafetySetting(
        category=gai.HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        threshold=gai.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]


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
        response = self._model.generate_content(
            self._input_text, safety_settings=safety_config
        )
        return response.text
