"""
Uses BERT and COCOAPI to preprocess the captions.

https://github.com/hanxiao/bert-as-service
This requires tensorflow >= 1.10
More information:
https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

Before running this file,
start BERT server first:

bert-serving-start -model_dir ../data/bert_models/uncased_L-12_H-768_A-12/ -max_seq_len NONE

TO-DO
- [] Add START and END sentence tags (do we need to?)
- [] Add POS tags (optional, after we have a working model already)
"""
from re import sub

import numpy as np

from pycocotools.coco import COCO
from bert_serving.client import BertClient
from settings import *


bert_client = BertClient()


print('''
    This program takes all the image captions for an image and calculates
    the mean of their fixed-length vector forms using Google's BERT.
    This may take a while.
    ''')

max_words = 0
for split in ['train', 'val']:
    print('Processing captions for {} split.'.format(split))

    # initialize all captions for the split using COCO API
    all_captions = COCO(CAPTIONS_DIR.format(split))

    # using the image ids from karpathy split
    with open(KARPATHY_SPLIT_DIR.format(split), 'r') as f:

        # get all image captions and calculate their mean fixed-length vector
        for i, img_id in enumerate(f.read().split('\n')[:-1]):
            caption_ids = all_captions.getAnnIds(imgIds=int(img_id))
            captions = [sub(r'[^\w ]', '', c['caption'].lower()).strip()
                        for c in all_captions.loadAnns(caption_ids)]

            # for data exploration:
            # check how many words are in the longest caption
            longest_caption = max([len(c.split()) for c in captions])
            if longest_caption > max_words:
                max_words = longest_caption

            # this returns an array of shape (5, 768)
            bert_vectors = bert_client.encode(captions)
            mean_vectors = np.mean(bert_vectors, 0)

            np.save(MEAN_VEC_DIR.format(img_id), mean_vectors)

            if i % 5000 == 0:
                print('Captions for {} images are finished.'.format(i))

    print('Finished processing {} split.'.format(split))
    print('Longest caption found: {} words'.format(max_words))
