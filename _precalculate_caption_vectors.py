"""
Uses BERT and cocoAPI to preprocess the captions.

https://github.com/hanxiao/bert-as-service
This requires tensorflow >= 1.10
More information:
https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html

# Run this first if using BERT encodings:
# bert-serving-start -model_dir ../data/bert_models/uncased_L-12_H-768_A-12/ -max_seq_len 30
"""
from re import sub

import numpy as np

from pycocotools.coco import COCO
from bert_serving.client import BertClient
from settings import *

bert_client = BertClient(check_length=False)

for split in ['train', 'val']:
    print('Processing captions for {} split.'.format(split))

    # initialize all captions for the split using coco API
    coco = COCO(CAPTIONS_DIR.format(split))
    if split == 'train':  # combine train and val
        val_coco = COCO(CAPTIONS_DIR.format('val'))
        coco.anns.update(val_coco.anns)
        coco.imgs.update(val_coco.imgs)
        coco.imgToAnns.update(val_coco.imgToAnns)
    id_captions_map = {}

    # using the image ids from karpathy split
    with open(KARPATHY_SPLIT_DIR.format(split), 'r') as f:

        # get all image captions and calculate their mean fixed-length vector
        for i, img_id in enumerate(f.read().split('\n')[:-1]):
            img_id = img_id.split()[-1]
            caption_ids = coco.getAnnIds(imgIds=int(img_id))
            captions = [sub(r'[^\w ]', '', c['caption'].lower()).strip()
                        for c in coco.loadAnns(caption_ids)]

            # this returns an array of shape (5, 768)
            bert_vectors = bert_client.encode(captions)
            id_captions_map[img_id] = bert_vectors

            if i % 5000 == 0:
                print('Captions for {} images are finished.'.format(i))

        np.save(CAPTION_VECTORS_DIR.format(split), id_captions_map)

    print('Finished processing {} split.'.format(split))
