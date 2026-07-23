# PyTorch Sandbox

Just a test repo for PyTorch.

## Set up

[Install PyTorch locally](https://pytorch.org/get-started/locally/)

## The Amstrad model

This is a model trained on facts about the Amstrad PCW range of personal computers.

Install [Hugging Face](https://huggingface.co/) libraries:

`pip3 install transformers datasets`

Install other libraries:

`pip3 install accelerate peft trl bitsandbytes sentencepiece`

Run the training script:

`python3 amstrad-train.py`

When finished, run:

`python3 amstrad-run.py`
