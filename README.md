# Learning Rational Subgoals from Demonstrations and Instructions

This repository is the implementation of "Learning Rational Subgoals from Demonstrations and Instructions" published at AAAI 2023.

## Requirements

Environment has been specified by `rsg.yml`

## Training

To train the model(s) in the paper, run this command:
```bash
./scripts/train_all.sh
```

## Evaluation

To evaluate the models (including classification, and planning with instructions/final goal), run:

```eval
./scripts/eval_all.sh
```

## Datasets

The datasets will be automatically generated when training models. To force re-generate datasets, add argument `--force_regen` in `scripts/train_all.py` or `scripts/eval_all.py` when calling `projects/rsg/scripts/learn_classifier.py`. 

