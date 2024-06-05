# How consistently can LLMs apply linguistic metrics?

Large language models are useful in a variety of tasks, including style transfer.
The question is, though, if we can objectively evaluate their usefulness.
Text style is defined by many attributes and the field of study that studies them
is called computational linguistics.
Many text style metrics focus on meaning preservation (tone and emotion, for
example) while others focus on syntactical structure (length of sentence of number
of exclamatory statements, for example).

We want to know if large language models can consistently follow instructions about
the latter category of metrics.

## Running the Experiment

Before you do anything, set up [`poetry`](https://python-poetry.org) on your
machine.

Next, run the LLM consistency `evaluator`. It will generate data based on
2 datasets:

* Yelp reviews
* CNN/DailyMail

```shell
poetry install
poetry run python -m llm_tst_consistency.evaluator 
```

The text generation is performed via 5 well-known LLMs. 

