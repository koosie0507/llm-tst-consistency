from dataclasses import asdict
from math import isnan, erf
from pathlib import Path

import numpy as np
import orjson as json
import pandas as pd
import spacy
from datasets import load_dataset, Dataset
from jinja2 import Environment, FileSystemLoader, select_autoescape
from spacy import Language

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


INPUT = """
Young people with internet addiction experience changes in their brain chemistry
which could lead to more addictive behaviours, research suggests.

The study, published in PLOS Mental Health, reviewed previous research using
functional magnetic resonance imaging (fMRI) to examine how regions of the brain
interact in people with internet addiction.

They found that the effects were evident throughout multiple neural networks in
the brains of young people, and that there was increased activity in parts of
the brain when participants were resting.

At the same time, there was an overall decrease in the functional connectivity
in parts of the brain involved in active thinking, which is the executive
control network of the brain responsible for memory and decision-making.

The research found that these changes resulted in addictive behaviours and
tendencies in adolescents, as well as behavioural changes linked to mental
health, development, intellectual ability and physical coordination.

The researchers reviewed 12 previous studies involving 237 10- to 19-year-olds
with a formal diagnosis of internet addiction between 2013 and 2023.

Almost half of British teenagers have said they feel addicted to social media,
according to a survey this year.

Max Chang, the study’s lead author and an MSc student at the UCL Great Ormond
Street Institute of Child Health (GOS ICH), said: “Adolescence is a crucial
developmental stage during which people go through significant changes in their
biology, cognition and personalities.

“As a result, the brain is particularly vulnerable to internet addiction-related
urges during this time, such as compulsive internet usage, cravings towards
usage of the mouse or keyboard and consuming media.

“The findings from our study show that this can lead to potentially negative
behavioural and developmental changes that could impact the lives of
adolescents. For example, they may struggle to maintain relationships and social
activities, lie about online activity and experience irregular eating and
disrupted sleep.”

Chang added that he hoped the findings demonstrated “how internet addiction
alters the connection between the brain networks in adolescence”, which would
then allow early signs of internet addiction to be treated effectively.

He added: “Clinicians could potentially prescribe treatment to aim at certain
brain regions or suggest psychotherapy or family therapy targeting key symptoms
of internet addiction.

“Importantly, parental education on internet addiction is another possible
avenue of prevention from a public health standpoint. Parents who are aware of
the early signs and onset of internet addiction will more effectively handle
screen time, impulsivity, and minimise the risk factors surrounding internet
addiction.”

Irene Lee, a senior author of the research paper also based at GOS ICH, said:
“There is no doubt that the internet has certain advantages. However, when it
begins to affect our day-to-day lives, it is a problem.

“We would advise that young people enforce sensible time limits for their daily
internet usage and ensure that they are aware of the psychological and social
implications of spending too much time online.”
"""


def _compute_dataset_features(nlp: Language, ds: Dataset) -> dict[str, Stats]:
    corpus = [row["article"] for row in ds][:10]
    docs = list(map(nlp, corpus))
    return {
        key: Stats.hlf(docs, HandcraftedLinguisticFeature(key))
        for key in HLF_TEMPLATES
    }


def _write_hlf_instructions(ds_stats: dict[str, Stats]) -> list[str]:
    result = []
    for feature, stats in ds_stats.items():
        template = load_template(HLF_TEMPLATES[feature])
        result.append(template.render(**asdict(stats)))
    return result


def areas_to_target(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    area_a = np.trapz(np.abs(np.array(a) - np.array(target)))
    area_b = np.trapz(np.abs(np.array(b) - np.array(target)))
    return area_a, area_b


def distance_norms_to_target(a: list[float], b: list[float], target: list[float]) -> tuple[float, float]:
    distance_a = np.linalg.norm(np.array(a) - np.array(target))
    distance_b = np.linalg.norm(np.array(b) - np.array(target))
    return distance_a, distance_b


def is_diff_significant(x: float, y: float, confidence: float = 0.95) -> bool:
    std_dev = np.std([x, y])
    z_score = abs(x - y) / std_dev

    # need to double-check this formula
    p_value = 1 - 0.5 * (1 + erf(z_score / np.sqrt(2)))
    if isnan(p_value):
        return False

    threshold = 1 - confidence
    return p_value < threshold


if __name__ == "__main__":
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
        baseline_generator = Ollama(INPUT, tpl.render(), HLFs)
        data = [baseline_generator("baseline") for _ in range(10)]

        # augment with hlf instructions
        hlf_generator = Ollama(INPUT, tpl.render(hlf_instructions=_write_hlf_instructions(ds_stats)), HLFs)
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
    for feature in features:
        target = [ds_stats[feature].mean]*len(df)
        a = df[f"baseline_{feature}"].values
        b = df[f"hlf_{feature}"].values

        dist_a, dist_b = distance_norms_to_target(a, b, target)
        euclid[feature] = dist_a - dist_b
        euclid[f"is_{feature}_significant"] = is_diff_significant(dist_a, dist_b)

        area_a, area_b = areas_to_target(a, b, target)
        area[feature] = area_a - area_b
        area[f"is_{feature}_significant"] = is_diff_significant(area_a, area_b)

    diff_df = pd.DataFrame(columns=list(euclid.keys()), data=[euclid, area])
