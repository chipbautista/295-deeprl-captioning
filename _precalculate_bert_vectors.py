from re import sub

import numpy as np

from settings import *
from pycocotools.coco import COCO
from bert_serving.client import BertClient


coco = COCO(CAPTIONS_DIR.format('train'))
val_coco = COCO(CAPTIONS_DIR.format('val'))

coco.anns.update(val_coco.anns)
coco.imgs.update(val_coco.imgs)
coco.imgToAnns.update(val_coco.imgToAnns)

bert_client = BertClient(check_length=False)
for split in ['train', 'val']:
    with open(KARPATHY_SPLIT_DIR.format(split)) as f:
        img_ids = f.read().split('\n')[:-1]

    img_ids = [int(i.split()[-1]) for i in img_ids]

    print('Going through {} captions and saving their BERT vectors...'.format(split))
    print('Total images: ', len(img_ids))

    for i, img_id in enumerate(img_ids):
        caption_ids = coco.getAnnIds(img_id)
        captions = coco.loadAnns(caption_ids)
        captions = [sub(r'[^\w ]', '', c['caption'].lower()).strip()
                    for c in captions]
        vectors = bert_client.encode(captions)
        np.save(CAPTION_VECTORS_DIR.format(split, img_id), vectors)

        if i % 5000 == 0:
            print(i, 'images processed.')
