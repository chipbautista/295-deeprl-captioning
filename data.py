"""
Should handle all data set processing tasks
for train and test time.
"""
import numpy as np
from re import sub
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from settings import *

"""
class MSCOCO(Dataset):
    def __init__(self, split, ):
        self.mode = mode
        # with open(KARPATHY_SPLIT_DIR.format(split)) as f:
        #     self.img_ids = f.read().split('\n')
        self.img_ids = LOCAL_IDS

    def __getitem__(self, index):
        return (
            np.load(FEATURES_DIR.format(self.img_ids[index])),  # shape (36, 2048)
            np.load(MEAN_VEC_DIR.format(self.img_ids[index]))  # shape (768,)
        )

    def __len__(self):
        return len(self.img_ids)
"""


class MSCOCO_Supervised(Dataset):
    def __init__(self, split):
        # self.img_ids = LOCAL_IDS
        with open(KARPATHY_SPLIT_DIR.format(split)) as f:
            self.img_ids = f.read().split('\n')[:-1][:10000]

        self.load_img_features()

        self.coco_captions = COCO(CAPTIONS_DIR.format(split))
        self.vocabulary = np.load('vocabulary.npy')

    def load_img_features(self):
        self.img_features = [np.load(FEATURES_DIR.format(img_id))
                             for img_id in self.img_ids]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        # load captions for this image
        caption_ids = self.coco_captions.getAnnIds(imgIds=int(img_id))
        # get a random caption
        caption_id = np.random.choice(caption_ids)
        caption = self.coco_captions.loadAnns([caption_id])[0]['caption']
        # preprocess caption into indeces of each word
        return (self.img_features[index], self._clean(caption) + ' <EOS>')

    def _clean(self, caption):
        return sub(r'[^\w ]', '', caption.lower()).strip()
