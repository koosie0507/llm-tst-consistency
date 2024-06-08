import os
from dataclasses import asdict
from enum import StrEnum
from functools import partial
from pathlib import Path

import dotenv
import orjson as json
import pandas as pd
import spacy
from datasets import Dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape
from spacy import Language

from llm_tst_consistency.dataset_loaders import load_cnn_daily_mail
from llm_tst_consistency.hlf import (
    HandcraftedLinguisticFeature,
    KupermanAgeOfAcquisition,
)
from llm_tst_consistency.llm import GPT, Gemini, Claude3, Ollama
from llm_tst_consistency.plot import draw_plots
from llm_tst_consistency.reporting import make_report
from llm_tst_consistency.stats import Stats


dotenv.load_dotenv()
MAX_CORPUS_SIZE = os.getenv("LTC_MAX_CORPUS_SIZE", 1)
TRIAL_COUNT = os.getenv("LTC_TRIAL_COUNT", 3)
HLF_TEMPLATES = {
    "t_word": "total_words.j2",
    "t_sent": "total_sentences.j2",
    "n_uverb": "number_of_unique_verbs.j2",
    "n_uadj": "number_of_unique_adjectives.j2",
    "simp_ttr": "simple_type_token_ratio.j2",
    "a_verb_pw": "avg_verbs_per_word.j2",
    "corr_adj_var": "corrected_adjective_variation.j2",
    "corr_verb_var": "corrected_verb_variation.j2",
    "fkgl": "grade_level.j2",
    "a_kup_pw": "kuperman_age.j2",
}
HLFs = {
    name: (
        KupermanAgeOfAcquisition(name)
        if name == "a_kup_pw"
        else HandcraftedLinguisticFeature(name)
    )
    for name in HLF_TEMPLATES
}

LLMs = {
    "claude3": partial(
        Claude3,
        model_name="claude-3-opus-20240229",
        api_key=os.getenv("LTC_ANTHROPIC_KEY"),
        hlf_cfg=HLFs,
    ),
    "gemini": partial(
        Gemini,
        project_name=os.getenv("LTC_GOOGLE_PROJECT"),
        location=os.getenv("LTC_GOOGLE_LOCATION"),
        hlf_cfg=HLFs,
        model_name="gemini-1.5-pro",
    ),
    "gpt": partial(
        GPT,
        model_name="gpt-4o",
        api_key=os.getenv("LTC_OPENAI_KEY"),
        hlf_cfg=HLFs,
    ),
    "llama3_8b": partial(
        Ollama,
        model_name="llama3",
        host=os.getenv("LTC_OLLAMA_HOST"),
        hlf_cfg=HLFs,
    ),
    "mixtral_8x7b": partial(
        Ollama,
        model_name="mixtral:8x7b",
        host=os.getenv("LTC_OLLAMA_HOST"),
        hlf_cfg=HLFs,
    ),
}
CURRENT_LLM = "llama3_8b"

DATASETS = {
    "cnn_dailymail": load_cnn_daily_mail,
}
CURRENT_DATASET = "cnn_dailymail"


class MetricLevel(StrEnum):
    BASELINE = "baseline"
    HLF = "hlf"


def _load_input_text(ds_name):
    fpath = Path(__file__).parent.parent.parent / "data" / f"{ds_name}_input.txt"
    if fpath.exists() and fpath.is_file():
        return fpath.read_text()
    raise ValueError(f"could not find input file associated with dataset", ds_name)


def _load_j2(template_name):
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir), autoescape=select_autoescape()
    )
    return env.get_template(template_name)


def _compute_dataset_features(nlp: Language, ds: Dataset) -> dict[str, Stats]:
    corpus = [row["article"] for row in ds][:MAX_CORPUS_SIZE]
    docs = list(map(nlp, corpus))
    return {name: Stats.hlf(docs, feature) for name, feature in HLFs.items()}


def _load_dataset_stats(ds_name, ds_loader):
    stats_file_name = Path(__file__).parent / f"{ds_name}.json"
    if not stats_file_name.exists():
        ds_stats = _compute_dataset_features(spacy.load("en_core_web_sm"), ds_loader())
        stats_file_name.write_bytes(json.dumps(ds_stats))
    else:
        ds_stats = json.loads(stats_file_name.read_text())
        for key in ds_stats:
            ds_stats[key] = Stats(**ds_stats[key])
    return ds_stats


def _write_hlf_instructions(ds_stats: dict[str, Stats]) -> list[str]:
    result = []
    for feature, stats in ds_stats.items():
        template = _load_j2(HLF_TEMPLATES[feature])
        result.append(template.render(**asdict(stats)))
    return result


def _generate_text_metrics(
    trial_count, prompt_template, hlf_instructions, input_text, model_cls, features
):
    tpl = _load_j2(prompt_template)

    # generate baseline
    baseline_prompt = tpl.render()
    baseline_generator = model_cls(
        text=input_text, prompt=baseline_prompt, metric_level=MetricLevel.BASELINE
    )
    data = [baseline_generator() for _ in range(trial_count)]

    # augment with hlf instructions
    hlf_prompt = tpl.render(hlf_instructions=hlf_instructions)
    hlf_generator = model_cls(
        text=input_text, prompt=hlf_prompt, metric_level=MetricLevel.HLF
    )
    for obj in data:
        obj.update(hlf_generator())

    return pd.DataFrame(
        data=data,
        columns=[
            f"{metric_level}_{feature}"
            for metric_level in MetricLevel
            for feature in features
        ],
    )


def _load_generated_text_metrics(model_family, ds_name, generate_text_cb):
    file_name = Path(__file__).parent / f"gen_results_{model_family}_{ds_name}.csv"
    if not file_name.exists():
        df = generate_text_cb()
        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name)
    return df


def main():
    # select the model
    if CURRENT_LLM not in LLMs:
        exit(2)
    model_class = LLMs[CURRENT_LLM]
    if CURRENT_DATASET not in DATASETS:
        exit(1)
    system_prompt_template = "prompt_1.j2"

    features = list(HLFs)
    ds_stats = _load_dataset_stats(CURRENT_DATASET, DATASETS[CURRENT_DATASET])
    input_text = _load_input_text(CURRENT_DATASET)
    hlf_instructions = _write_hlf_instructions(ds_stats)
    df = _load_generated_text_metrics(
        CURRENT_LLM,
        CURRENT_DATASET,
        lambda: _generate_text_metrics(
            TRIAL_COUNT,
            system_prompt_template,
            hlf_instructions,
            input_text,
            model_class,
            features,
        ),
    )
    draw_plots(CURRENT_LLM, df, features, ds_stats)
    report = make_report(features, ds_stats, df)
    report.to_csv(f"report_{CURRENT_LLM}_{CURRENT_DATASET}.csv")


if __name__ == "__main__":
    main()
