"""
Should handle all data set processing tasks
for train and test time.
"""
import numpy as np
from re import sub
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from settings import *


class MSCOCO(Dataset):
    def __init__(self, split, evaluation=False):
        # Set evaluation to True if you want this to return
        # all the captions for a single image.
        # 1 sample = [image, 5 captions]
        # If false, then pairs each image with each of its captions
        # 1 sample = [image, 1 caption]
        self.evaluation = evaluation
        with open(KARPATHY_SPLIT_DIR.format(split)) as f:
            self.img_ids = f.read().split('\n')[:-1]

        self.coco = COCO(CAPTIONS_DIR.format(split))

        if not self.evaluation and PAIR_ALL_CAPTIONS:
            # Pair images to all its ground truth captions
            # this results to 414,113 image-caption pairs
            self.pair_all_captions()

    def pair_all_captions(self):
        self.img_caption_pairs = []
        for img_id in self.img_ids:
            caption_ids = self.coco.getAnnIds(imgIds=int(img_id))
            captions = [self._preprocess(c['caption']) for c in
                        self.coco.loadAnns(caption_ids)]
            self.img_caption_pairs.extend([
                (img_id, caption) for caption in captions
            ])

    def __getitem__(self, index):
        # Case 1: When we give out a total of 414,113 image-caption pairs
        # (For supervised training)
        if PAIR_ALL_CAPTIONS and not self.evaluation:
            img_id, caption = self.img_caption_pairs[index]
            return (np.load(FEATURES_DIR.format(img_id)), caption)

        img_id = self.img_ids[index]
        img_features = np.load(FEATURES_DIR.format(img_id))
        caption_ids = self.coco.getAnnIds(imgIds=int(img_id))

        # Case 2: When we're doing RL training and want to get all the
        # image's captions. Also for evaluating caption score.
        if self.evaluation:
            return (img_id,
                    img_features,
                    [self._preprocess(c['caption'])
                     for c in self.coco.loadAnns(caption_ids)]
                    )

        # Case 3: When we just randomly choose a caption for each image
        caption_id = np.random.choice(caption_ids)
        caption = self._preprocess(
            self.coco.loadAnns([caption_id])[0]['caption'])
        return (img_features, caption)

    def __len__(self):
        # Case 1
        if PAIR_ALL_CAPTIONS and not self.evaluation:
            return len(self.img_caption_pairs)

        # For Case 2 and Case 3
        return len(self.img_ids)

    def _preprocess(self, caption):
        # Basically removes non-alphanumeric characters,
        # converts to undercase, and adds <EOS>
        return ' '.join([sub(r'[^\w ]', '', caption.lower()).strip(), '<EOS>'])
