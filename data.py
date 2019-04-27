"""
Should handle all data set processing tasks
for train and test time.
"""

from torch.utils.data import Dataset
from settings import *


class MSCOCO(Dataset):
    def __init__(self, split):
        with open(KARPATHY_SPLIT_DIR.format(split)) as f:
            self.img_ids = f.read().split('\n')

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        return (
            np.load(FEATURES_DIR.format(img_id)),  # image features
            np.load(MEAN_VEC_DIR.format(img_id))  # mean of caption embeddings
        )

    def __len__(self):
        return len(self.img_ids)
