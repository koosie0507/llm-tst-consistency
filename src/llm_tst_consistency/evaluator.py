import os.path
from dataclasses import asdict
from pathlib import Path

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

