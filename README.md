# How consistently can LLMs apply linguistic metrics?

Large language models are useful in a variety of tasks, including text style
transfer.
The question is, though, how might we evaluate their usefulness.
There's already lots of answers to this question and all of them come with
caveats.
Albeit not a perfect answer to this question, handcrafted linguistic features
are:

* a well documented notion in computational linguisitics
* able to evaluate text style (at least partially).

HLFs have received recent attention in the context of assessing text
readability.
This repository contains an experiment that assesses how well large language
models can follow instructions derived from HLFs.

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

To run the experiment code use the following command.

```shell
poetry run python -m llm_tst_consistency.evaluator 
```

To run the experiment with the parameters used in the accompanying paper, use
the `./run_experiment.sh` shell script.

You may configure the input text used for text generation for each dataset by
changing the contents of the corresponding files in the [`./data`](./data)
directory.
