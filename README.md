# How consistently can LLMs apply linguistic metrics?

Large language models are useful in a variety of tasks, including text style
transfer.
The question is, though, how might we evaluate their usefulness.
There's already lots of answers to this question and all of them come with
caveats.
Far from being a perfect answer, handcrafted linguistic features can evaluate at
least partially whether one text style matches another.

HLFs are well documented in computational linguistics and have received recent
attention in the context of assessing text readability.
This reposiory contains an experiment that assesses how well large language
models can follow instructions derived from handcrafted linguistic features.

## Running the Experiment

1. Set up [`poetry`](https://python-poetry.org)
1. Perform the individual prerequisite setup steps for each LLM in the
experiment.
    * set up API accounts for OpenAI (`gpt-4o`), Google (`gemini-1.5-pro`) and
    Anthropic (`claude3-opus`);
    * set up [Ollama](https://ollama.com) to run Llama3 or Command-R+;
    * configure settings by creating a `.env` file based on the provided
    `.env.sample`.
1. Install the experiment in its own `virtualenv` by running `poetry install`

Now you're (hopefully) ready to run the LLM consistency `evaluator`. It will
generate data based on 2 datasets:

* Yelp reviews
* CNN/DailyMail

To run the experiment use the following command.

```shell
poetry run python -m llm_tst_consistency.evaluator 
```

You may configure the input text used for text generation for each dataset by
changing the contents of the corresponding files in the [`./data`](./data)
directory.
