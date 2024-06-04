import os.path
from dataclasses import asdict
from pathlib import Path

import orjson as json
import pandas as pd
import spacy
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature
from llm_tst_consistency.llm import Ollama
from llm_tst_consistency.plot import draw_plots
from llm_tst_consistency.stats import Stats


def load_template(name):
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape()
    )
    return env.get_template(name)


def is_cnn(record: dict) -> bool:
    return not record.get("article", "").upper().startswith("(CNN)")


def load_cnn_daily_mail():
    return load_dataset("abisee/cnn_dailymail", "3.0.0", split="test").filter(is_cnn)


HLF_TEMPLATES = {
    "t_uword": "total_unique_words.j2",
    "t_sent": "total_sentences.j2",
    "t_n_ent": "total_named_entities.j2",
    "n_uadj": "total_unique_adjectives.j2",
    "simp_ttr": "simple_type_token_ratio.j2",
    "a_verb_pw": "avg_verbs_per_word.j2",
    "corr_adj_var": "corrected_adjective_variation.j2",
    "corr_verb_var": "corrected_verb_variation.j2",
    "fkgl": "grade_level.j2",
    "a_kup_pw": "kuperman_age.j2",
}
HLFs = {name: HandcraftedLinguisticFeature(name) for name in HLF_TEMPLATES}


def render_hlf_instruction(template_name: str, stats: Stats) -> str:
    template = load_template(template_name)
    return template.render(**asdict(stats))


INPUT = "Interesting fact: by the year 2020 all actors on american tv shows will be australian."


if __name__ == "__main__":
    tpl = load_template("prompt_1.j2")
    nlp = spacy.load("en_core_web_sm")
    ds = load_cnn_daily_mail()
    corpus = [row["article"] for row in ds][:10]
    docs = list(map(nlp, corpus))
    ds_stats = {
        key: Stats.hlf(docs, HandcraftedLinguisticFeature(key))
        for key in HLF_TEMPLATES
    }
    instr = list(map(lambda x: render_hlf_instruction(HLF_TEMPLATES[x], ds_stats[x]), ds_stats))
    column_names = list(HLFs)

    baseline_csv = "baseline_llama_3.csv"
    if not os.path.exists(baseline_csv):
        baseline_prompt = tpl.render()
        generate_text = Ollama(INPUT, baseline_prompt, HLFs)
        stats_df = pd.DataFrame(data=[generate_text() for _ in range(10)], columns=column_names)
        stats_df.to_csv(baseline_csv)
    else:
        stats_df = pd.read_csv(baseline_csv)

    hlf_csv = "hlf_llama_3.csv"
    if not os.path.exists(hlf_csv):
        hlf_prompt = tpl.render(hlf_instructions=instr)
        generate_hlf_text = Ollama(INPUT, hlf_prompt, HLFs)
        hlf_stats_df = pd.DataFrame([generate_hlf_text() for _ in range(10)], columns=column_names)
        hlf_stats_df.to_csv(hlf_csv)
    else:
        hlf_stats_df = pd.read_csv(hlf_csv)

    draw_plots("llama 3 - baseline ", stats_df, column_names, ds_stats)
    draw_plots("llama 3 - hlf", hlf_stats_df, column_names, ds_stats)

