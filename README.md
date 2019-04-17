# 295-deeprl-captioning
A repository for our CS 295 (Reinforcement Learning) Mini Project

### Directory Structure
Create a `data` folder on the same level as this repo.
The folder should have three more folders inside:
1. `bert_models` - place the BERT models inside. [See here](https://github.com/hanxiao/bert-as-service#install)
2. `coco` - This is the folder for MSCOCO API. Follow instructions from [their repo](https://github.com/cocodataset/cocoapi). Captions are inside `annotations` folder, and image folders (`train2014` and `val2014`) should be inside `images`.
3. `karpathy_splits` - text files containing image IDs and/or filenames for MSCOCO train-test-val splits used commonly by researchers.
