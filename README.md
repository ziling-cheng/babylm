## McGill BabyLM Shared Task Submission: The Effects of Data Formatting and Structural Biases
This repository contains the code for the paper published in Proceedings of the BabyLM Challenge at the 27th CoNLL.

We also release the datasets and models on the [HuggingFace Hub](https://huggingface.co/mcgill-babylm).

### Pretraining Experiments
`run_bert.py` and `run_gpt.py` are the main files for pretraining models on BabyLM datasets, with helper functions defined in `utils.py`.

`configs` folder contains the configurations for training, model, and data.

`modeling` folder contains the architectural changes for the POS-augmented pretrained models.

`scripts` folder contains the scripts to pretrain the models.

### Citation
```
@inproceedings{cheng-etal-2023-mcgill,
    title = "{M}c{G}ill {B}aby{LM} Shared Task Submission: The Effects of Data Formatting and Structural Biases",
    author = "Cheng, Ziling  and
      Aralikatte, Rahul  and
      Porada, Ian  and
      Spinoso-Di Piano, Cesare  and
      Cheung, Jackie CK",
    booktitle = "Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning",
    year = "2023",
    url = "https://aclanthology.org/2023.conll-babylm.18"
}
```

