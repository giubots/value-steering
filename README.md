# Prompt-Based Value Steering of Large Language Models

This repository contains the code and resources for the paper "Prompt-Based Value Steering of Large Language Models", presented at the 3rd International Workshop on Value Engineering in AI (VALE 2025), 28th European Conference on AI, and to appear in Springer LNCS.

See [the notebook](value_steering_procedure.ipynb) for the complete procedure and an example.

## Abstract

Large language models are increasingly used in applications where alignment with human values is critical. While model fine-tuning is often employed to ensure safe responses, this technique is static and does not lend itself to everyday situations involving dynamic values and preferences. In this paper, we present a practical, reproducible, and model-agnostic procedure to evaluate whether a prompt candidate can effectively steer generated text toward specific human values, formalising a scoring method to quantify the presence and gain of target values in generated responses. We apply our method to a variant of the Wizard-Vicuna language model, using Schwartz's theory of basic human values and a structured evaluation through a dialogue dataset. With this setup, we compare a baseline prompt to one explicitly conditioned on values, and show that value steering is possible even without altering the model or dynamically optimising prompts.