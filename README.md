# 295-deeprl-captioning
A repository for our CS 295 (Reinforcement Learning) Mini Project. Partnered with Jakov Dumbrique.

### Dependencies
- [Install BERT server](https://github.com/hanxiao/bert-as-service#install): `pip install bert-serving-server` and `pip install bert-serving-client`
- [COCO API](https://github.com/cocodataset/cocoapi) or by `pip install pycocotools`
(install `Cython` first!)

**For ASTI's HPCC**
Latest TensorFlow version is 1.13, but HPCC only supports until 1.12 because of outdated CUDA softwares (I think).

Install TF 1.12.0 explicitly with:
`conda install -c anaconda tensorflow-gpu=1.12.0`

### File structure
1. Clone this repo: `git clone https://github.com/chippybautista/295-deeprl-captioning.git`
2. Create data folder in the same level: `mkdir data`
3. Create subfolder for the BERT model. Download the base, uncased model, and unzip:
```
    mkdir data/bert_models
    wget -P data/bert_models https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip data/bert_models/uncased_L-12_H-768_A-12.zip -d data/bert_models/
```
4.  Create subfolder for COCO files. Download the data set and unzip:
```
    mkdir data/coco
    mkdir data/coco/annotations
    mkdir data/coco/images

    # these will take a while!
    wget http://images.cocodataset.org/zips/train2014.zip
    wget http://images.cocodataset.org/zips/val2014.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip train2014.zip -d data/coco/images
    unzip val2014.zip -d data/coco/images
    unzip annotations_trainval2014.zip -d data/coco/annotations
```
5. Create subfolder for MSCOCO Karpathy splits:
```
    mkdir data/karpathy_splits
```
6a. Create subfolder for image features c/o [this repo/project from peteanderson08](https://github.com/peteanderson80/bottom-up-attention) and download 36 features per image:
```
    mkdir data/features
    # this is 25GB!
    wget https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
    unzip trainval_36.zip -d data/features/
```
6b. For easy access to the features during runtime, extract them from the `.tsv` file to a separate folder (will need 35GB disk space):
```
    mkdir data/features/extracts

    # run this using Python 2!
    python extract_features.py
```