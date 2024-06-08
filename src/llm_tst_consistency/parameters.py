from enum import StrEnum


class MetricLevel(StrEnum):
    BASELINE = "baseline"
    HLF = "hlf"


class LLMName(StrEnum):
    CLAUDE = "claude3"
    GEMINI = "gemini"
    GPT = "gpt"
    LLAMA3 = "llama3_8b"
    MIXTRAL = "mixtral_8x7b"


class DatasetName(StrEnum):
    CNN_DAILY_MAIL = "cnn_dailymail"
    YELP_REVIEWS = "yelp"


class PromptName(StrEnum):
    PROMPT1 = "prompt_1"
    PROMPT2 = "prompt_2"
