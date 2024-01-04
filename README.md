## McGill BabyLM Shared Task Submission: The Effects of Data Formatting and Structural Biases
This repository contains the code for the paper published in Proceedings of the BabyLM Challenge at the 27th CoNLL.

We also release the datasets used and models trained on the [HuggingFace Hub](https://huggingface.co/mcgill-babylm).

`run_bert.py` and `run_gpt.py` are the main files for pretraining models on BabyLM datasets, with helper functions defined in `utils.py`.
`configs` folder contains the configurations for training, model, and data.
`modeling` folder contains the architectural changes for the POS-augmented pre-trained models.
`scripts` folder contains the scripts to pretrain the models.



