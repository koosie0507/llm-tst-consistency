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
Each of the LLMs in this experiment has its own requirements. Make sure you meet
them. The models from OpenAI (`gpt-4o`), Google (`gemini-1.5-pro`) and
Anthropic (`claude3-opus`) require setting up accounts with those companies.
Please configure the required credentials and settings by creating a `.env` file
based on the provided `.env.sample`.
The Llama3 and Command-R+ models requires a running [Ollama](https://ollama.com)
server.
Configure your Ollama setup by changing the values of the `LTC_OLLAMA_*` vars.

Now you're (hopefully) ready to run the LLM consistency `evaluator`. It will
generate data based on 2 datasets:

* Yelp reviews
* CNN/DailyMail

```shell
poetry install
poetry run python -m llm_tst_consistency.evaluator 
```

The text generation is performed via 5 well-known LLMs.
You may configure the input text used for text generation for each dataset by
changing the contents of the corresponding files in the [`./data`](./data)
directory.
