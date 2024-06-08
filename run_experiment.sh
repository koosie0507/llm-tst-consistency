#!/bin/sh

# run all LLMs on all datasets with all prompts
poetry run evaluator -n 10 -s 1000 -e 3
# then run all LLMs on all datasets with input from the dataset, but only for prompt 2
poetry run evaluator -n 10 -s 1000 -e 3 -p prompt_2 --input-from-dataset