import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from random import sample

import dotenv
import orjson as json
import pandas as pd
import spacy
import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from spacy import Language

from llm_tst_consistency.dataset_loaders import load_cnn_daily_mail
from llm_tst_consistency.hlf import (
    HandcraftedLinguisticFeature,
    KupermanAgeOfAcquisition,
)
from llm_tst_consistency.llm import GPT, Gemini, Claude3, Ollama
from llm_tst_consistency.parameters import LLMName, MetricLevel, DatasetName, PromptName
from llm_tst_consistency.plot import draw_plots
from llm_tst_consistency.reporting import make_report
from llm_tst_consistency.stats import Stats


dotenv.load_dotenv()
cli = typer.Typer(name="evaluator")

HLF_TEMPLATE_CONFIG = {
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
    for name in HLF_TEMPLATE_CONFIG
}
LLM_CONFIG = {
    LLMName.CLAUDE: partial(
        Claude3,
        model_name="claude-3-opus-20240229",
        api_key=os.getenv("LTC_ANTHROPIC_KEY"),
        hlf_cfg=HLFs,
    ),
    LLMName.GEMINI: partial(
        Gemini,
        project_name=os.getenv("LTC_GOOGLE_PROJECT"),
        location=os.getenv("LTC_GOOGLE_LOCATION"),
        hlf_cfg=HLFs,
        model_name="gemini-1.5-pro",
    ),
    LLMName.GPT: partial(
        GPT,
        model_name="gpt-4o",
        api_key=os.getenv("LTC_OPENAI_KEY"),
        hlf_cfg=HLFs,
    ),
    LLMName.LLAMA3: partial(
        Ollama,
        model_name="llama3",
        host=os.getenv("LTC_OLLAMA_HOST"),
        hlf_cfg=HLFs,
    ),
}
DATASET_CONFIG = {
    DatasetName.CNN_DAILY_MAIL: load_cnn_daily_mail,
}


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


def _compute_dataset_features(nlp: Language, corpus: list[str]) -> dict[str, Stats]:
    docs = list(map(nlp, corpus))
    return {name: Stats.hlf(docs, feature) for name, feature in HLFs.items()}


def _process_dataset(ds_name, ds_loader, max_corpus_size, max_example_count):
    stats_file_name = Path(__file__).parent.parent.parent / "data" / f"{ds_name}_settings.json"
    if not stats_file_name.exists():
        corpus = ds_loader(max_corpus_size)
        obj = {"stats": _compute_dataset_features(spacy.load("en_core_web_sm"), corpus), "examples": list(sample(corpus, max_example_count))}
        stats_file_name.write_bytes(json.dumps(obj))
    else:
        obj = json.loads(stats_file_name.read_text())
        ds_stats = obj["stats"]
        for key in ds_stats:
            ds_stats[key] = Stats(**ds_stats[key])
    return obj["stats"], obj["examples"]


def _write_hlf_instructions(ds_stats: dict[str, Stats]) -> list[str]:
    result = []
    for feature, stats in ds_stats.items():
        template = _load_j2(HLF_TEMPLATE_CONFIG[feature])
        result.append(template.render(**asdict(stats)))
    return result


def _generate_text_metrics(
    trial_count, prompt_template, hlf_instructions, input_text, model_cls, features, examples=None):
    tpl = _load_j2(prompt_template)

    # generate baseline
    example_text = f"{os.linesep}---{os.linesep}".join(examples)
    baseline_prompt = tpl.render(examples=example_text)
    baseline_generator = model_cls(
        text=input_text, prompt=baseline_prompt, metric_level=MetricLevel.BASELINE
    )
    data = [baseline_generator() for _ in range(trial_count)]

    # augment with hlf instructions
    hlf_prompt = tpl.render(hlf_instructions=hlf_instructions, examples=example_text)
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


def _load_generated_text_metrics(prompt_name, model_family, ds_name, generate_text_cb):
    file_name = Path(__file__).parent / f"gen_results_{prompt_name}_{model_family}_{ds_name}.csv"
    if not file_name.exists():
        df = generate_text_cb()
        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name)
    return df


@cli.command()
def main(
    llms: list[LLMName] = typer.Option(
        list(LLM_CONFIG), "-l", "--llm", help="run experiment with these LLM(s)"
    ),
    ds_names: list[DatasetName] = typer.Option(
        list(DATASET_CONFIG), "-d", "--dataset", help="run experiment on these datasets"
    ),
    prompt_names: list[PromptName] = typer.Option(
        list(PromptName), "-p", "--prompt", help="run experiment using this prompt"
    ),
    trial_count: int = typer.Option(
        10,
        "-n",
        "--count",
        help="the number of times we should generate text",
        envvar="LTC_TRIAL_COUNT",
    ),
    max_size: int = typer.Option(
        100,
        "-s",
        "--dataset-size",
        help="the maximum number of items we should take from the dataset",
        envvar="LTC_MAX_CORPUS_SIZE",
    ),
    example_count: int = typer.Option(
        1,
        "-e",
        "--example-count",
        help="the maximum number of examples to use in the prompts that make use of them",
        envvar="LTC_EXAMPLE_COUNT",
    ),
):
    for prompt_name in prompt_names:
        report = pd.DataFrame()
        for llm in llms:
            for ds_name in ds_names:
                # read input text corresponding to dataset
                input_text = _load_input_text(ds_name)
                # compute/load dataset stats
                ds_stats, examples = _process_dataset(ds_name, DATASET_CONFIG[ds_name], max_corpus_size=max_size, max_example_count=example_count)
                # create prompt based on dataset stats
                hlf_instructions = _write_hlf_instructions(ds_stats)
                # load text generation results / run text generation and compute results
                features = list(HLFs)
                df = _load_generated_text_metrics(
                    prompt_name,
                    llm,
                    ds_name,
                    lambda: _generate_text_metrics(
                        trial_count,
                        f"{prompt_name}.j2",
                        hlf_instructions,
                        input_text,
                        LLM_CONFIG[llm],
                        features,
                        examples,
                    )
                )
                #
                draw_plots(prompt_name, f"{prompt_name}-{llm}-{ds_name}", df, features, ds_stats)
                individual_report = make_report(llm, ds_name, features, ds_stats, df)
                report = pd.concat([report, individual_report], axis=0)
        report.to_csv(f"report_{prompt_name}.csv")


if __name__ == "__main__":
    cli()
