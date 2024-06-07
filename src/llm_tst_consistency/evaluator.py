from dataclasses import asdict
from math import isnan, erf
from pathlib import Path

import dotenv
import numpy as np
import orjson as json
import pandas as pd
import spacy
from datasets import load_dataset, Dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy.stats import ks_2samp
from spacy import Language

from llm_tst_consistency.hlf import HandcraftedLinguisticFeature, KupermanAgeOfAcquisition
from llm_tst_consistency.llm import OpenAI, Gemini
from llm_tst_consistency.plot import draw_plots
from llm_tst_consistency.stats import Stats


MAX_CORPUS_SIZE = 100


def load_template(name):
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape()
    )
    return env.get_template(name)


def is_cnn(record: dict) -> bool:
    return "(CNN)" in record.get("article", "")


def load_cnn_daily_mail():
    return load_dataset(
        "abisee/cnn_dailymail", "3.0.0", split="test"
    ).filter(is_cnn)


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
    name: KupermanAgeOfAcquisition(name) if name == "a_kup_pw" else HandcraftedLinguisticFeature(name)
    for name in HLF_TEMPLATES
}


INPUT = """I've been to this cafe a few times. The service is great and the menu
is affordable. It has a large spacious space, reliable free wifi, and good food.

It is quite new and seems a bit confused with which demographic of clienteles
they are catering to. It has a great potential to be a hub for creative, hipster
crowd.

At the moment, it is underrated. Thus, I give them an extra star on top of the 4
stars they deserve.
"""


def _compute_dataset_features(nlp: Language, ds: Dataset) -> dict[str, Stats]:
    corpus = [row["article"] for row in ds][:MAX_CORPUS_SIZE]
    docs = list(map(nlp, corpus))
    return {
        name: Stats.hlf(docs, feature)
        for name, feature in HLFs.items()
    }


def _write_hlf_instructions(ds_stats: dict[str, Stats]) -> list[str]:
    result = []
    for feature, stats in ds_stats.items():
        template = load_template(HLF_TEMPLATES[feature])
        result.append(template.render(**asdict(stats)))
    return result


def areas(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    area_a = np.trapz(np.abs(np.array(a) - np.array(target)))
    area_b = np.trapz(np.abs(np.array(b) - np.array(target)))
    return area_a, area_b


def euclid_norms(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    distance_a = np.linalg.norm(np.array(a) - np.array(target))
    distance_b = np.linalg.norm(np.array(b) - np.array(target))
    return distance_a, distance_b


def is_diff_significant(x: float, y: float, confidence: float = 0.95) -> bool:
    c = abs(x - y)
    std_dev = np.std([x, y])
    mu = np.mean([x, y])
    z_score = c - mu / std_dev

    # need to double-check this formula
    p_value = 1 - 0.5 * (1 + erf(z_score / np.sqrt(2)))
    if isnan(p_value):
        return False

    threshold = 1 - confidence
    return p_value < threshold


if __name__ == "__main__":
    dotenv.load_dotenv()
    file_name = Path(__file__).parent / "data.csv"
    features = list(HLFs)
    tpl = load_template("prompt_1.j2")
    stats_file_name = Path(__file__).parent / "cnn_dailymail.json"
    if not stats_file_name.exists():
        ds_stats = _compute_dataset_features(spacy.load("en_core_web_sm"), load_cnn_daily_mail())
        stats_file_name.write_bytes(json.dumps(ds_stats))
    else:
        ds_stats = json.loads(stats_file_name.read_text())
        for key in ds_stats:
            ds_stats[key] = Stats(**ds_stats[key])
    if not file_name.exists():
        # generate baseline
        baseline_generator = Gemini(INPUT, tpl.render(), HLFs)
        data = [baseline_generator("baseline") for _ in range(10)]

        # augment with hlf instructions
        hlf_generator = Gemini(INPUT, tpl.render(hlf_instructions=_write_hlf_instructions(ds_stats)), HLFs)
        for obj in data:
            obj.update(hlf_generator("hlf"))

        # create dataframe
        df = pd.DataFrame(data=data, columns=[
            f"{metric_set}_{feature}"
            for metric_set in ["baseline", "hlf"]
            for feature in features
        ])
        df.to_csv(file_name)
    else:
        df = pd.read_csv(file_name)

    draw_plots("llama 3", df, features, ds_stats)
    euclid = {}
    area = {}
    ks = {}
    for feature in features:
        diff_col = f"{feature}_diff"
        is_closer_col = f"is_{feature}_closer"
        is_different_col = f"is_{feature}_different"
        target = [ds_stats[feature].mean]*len(df)
        baseline = df[f"baseline_{feature}"].values
        hlf = df[f"hlf_{feature}"].values

        # Kolmogorov-Smirnov test
        ks_bh = ks_2samp(baseline, hlf)
        ks_bt = ks_2samp(baseline, target)
        ks_ht = ks_2samp(hlf, target)
        ks[diff_col] = ks_bt.statistic - ks_ht.statistic
        ks[is_closer_col] = ks_bt.statistic > ks_ht.statistic
        ks[is_different_col] = ks_bh.pvalue < 0.25

        # Euclidian distance
        dist_a, dist_b = euclid_norms(baseline, hlf, target)
        euclid[diff_col] = dist_a - dist_b
        euclid[is_closer_col] = dist_a > dist_b
        euclid[is_different_col] = is_diff_significant(dist_a, dist_b)

        # Area surface to target
        area_a, area_b = areas(baseline, hlf, target)
        area[diff_col] = area_a - area_b
        area[is_closer_col] = area_a > area_b
        area[is_different_col] = is_diff_significant(area_a, area_b)

    diff_df = pd.DataFrame(columns=list(euclid.keys()), data=[euclid, area, ks], index=["by euclidian norm", "by area", "k-s test"])
    diff_df.to_csv("report.csv")
