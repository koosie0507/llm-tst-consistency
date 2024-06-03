from dataclasses import asdict
from pathlib import Path

import lftk
import spacy
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape
from spacy.tokens.doc import Doc

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


HLF_MAP = {
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


def render_hlf_instruction(template_name: str, stats: Stats) -> str:
    template = load_template(template_name)
    return template.render(**asdict(stats))


class HandcraftedLinguisticFeature:
    def __init__(self, name: str):
        self._name = name

    def __call__(self, doc: Doc) -> float:
        extractor = lftk.Extractor(docs=doc)
        return extractor.extract(features=[self._name])[self._name]


if __name__ == "__main__":
    tpl = load_template("prompt_1.j2")
    nlp = spacy.load("en_core_web_sm")
    ds = load_cnn_daily_mail()
    corpus = [row["article"] for row in ds][:10]
    docs = list(map(nlp, corpus))
    ds_stats = {
        HLF_MAP[key]: Stats.hlf(docs, HandcraftedLinguisticFeature(key))
        for key in HLF_MAP
    }
    instr = list(map(lambda x: render_hlf_instruction(*x), ds_stats.items()))
    print(tpl.render())
    print("----------------------------------------------------------------")
    print(tpl.render(hlf_instructions=instr))
