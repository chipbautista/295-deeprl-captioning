"""
Uses BERT and COCOAPI to preprocess the captions.

https://github.com/hanxiao/bert-as-service
This requires tensorflow >= 1.10
More information:
https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

Before running this file,
start BERT server first:

bert-serving-start -model_dir ../data/bert_models/uncased_L-12_H-768_A-12/

TO-DO
- [] Add START and END sentence tags
- [] Add POS tags (optional, after we have a working model already)
"""
import json
import numpy as np

from pycocotools.coco import COCO
from bert_serving.client import BertClient


bert_client = BertClient()


print('''
    This program takes all the image captions for an image and calculates
    the mean of their fixed-length vector forms using Google's BERT.
    This may take a while.
    ''')

SPLIT_IDS_DIR = '../data/karpathy_splits/coco2014_cocoid.{}.txt'
CAPTIONS_DIR = '../data/coco/annotations/instances_{}2014.json'
MEAN_VEC_DIR = '../data/karpathy_splits/mean_bert_vectors.{}.txt'

for split in ['train', 'val']:
    print('Processing captions for {} split.'.format(split))
    mean_vectors = {}

    # initialize all captions for the split using COCO API
    all_captions = COCO(CAPTIONS_DIR.format(split))

    # using the image ids from karpathy split
    with open(SPLIT_IDS_DIR.format(split), 'r') as f:

        # get all image captions and calculate their mean fixed-length vector
        for i, img_id in enumerate(f.readline()):
            caption_ids = all_captions.getAnnIds(imgIds=img_id)
            captions = all_captions.loadAnns(caption_ids)

            # this returns an array of shape (5, 768)
            bert_vectors = bert_client.encode([c['caption'] for c in captions])
            mean_vectors[img_id] = np.mean(bert_vectors, 0)

            if i % 5000 == 0:
                print('Captions for {} images are finished.'.format(i))

    # save dictionary to file
    with open(MEAN_VEC_DIR.format(split), 'r+') as f:
        f.write(json.dumps(mean_vectors))
    print('Finished processing {} split.'.format(split))
