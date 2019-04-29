"""
Should handle all data set processing tasks
for train and test time.
"""
import numpy as np
from re import sub
from torch.utils.data import Dataset
from pycocotools.coco import COCO

from settings import *

# Because I can't download the whole 25GB file to my local PC,
# I copied some extracted image features from ASTI.
# These are only for testing! Total of 161 training features.
# LOCAL_IDS = ['123008',  '123013',  '12302',  '123023',  '123028',  '123036',  '123038',  '123041',  '123055',  '123066',  '123067',  '12307',  '123074',  '123083',  '123099',  '123117',  '123127',  '12313',  '123142',  '123147',  '12315',  '123155',  '123166',  '123172',  '123175',  '123177',  '123180',  '123184',  '123188',  '123190',  '123193',  '1232',  '123201',  '123208',  '12321',  '123214',  '123229',  '123247',  '123260',  '123262',  '123268',  '123269',  '123273',  '123282',  '123286',  '123298',  '123306',  '123310',  '123312',  '123330',  '123333',  '123336',  '123351',  '123359',  '123366',  '123371',  '123378',  '123382',  '123389',  '123411',  '123412',  '123413',  '123421',  '123424',  '123444',  '123445',  '12345',  '123453',  '123456',  '123457',  '123462',  '123473',  '12349',  '123509',  '123512',  '123514',  '123523',  '123532',  '123535',  '123539',  '123544',  '123552',  '123558',  '123568',  '12357',  '123579',  '123582',  '123612',  '123614',  '123620',  '123643',  '123646',  '123647',  '123664',  '123674',  '123692',  '123694',  '123697',  '123699',  '1237',  '123711',  '123719',  '12372',  '123721',  '123722',  '123731',  '123739',  '123749',  '123751',  '123753',  '123757',  '123762',  '123764',  '123765',  '123766',  '12377',  '123770',  '123776',  '123777',  '123788',  '12379',  '123791',  '123797',  '123799',  '1238',  '123800',  '12382',  '123824',  '123831',  '123835',  '123837',  '123841',  '123848',  '123851',  '123855',  '12386',  '123862',  '123878',  '123891',  '123892',  '123901',  '123907',  '123909',  '123914',  '123916',  '123920',  '123921',  '123923',  '123935',  '123938',  '123939',  '123944',  '123949',  '123967',  '123968',  '123970',  '123974',  '123975',  '12398',  '123980',  '123995']


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


class MSCOCO_Supervised(Dataset):
    def __init__(self, split):
        # self.img_ids = LOCAL_IDS
        with open(KARPATHY_SPLIT_DIR.format(split)) as f:
            self.img_ids = f.read().split('\n')[:-1]
        self.coco_captions = COCO(CAPTIONS_DIR.format(split))

        self.img_caption_pairs = []
        self.make_img_caption_pairs()

    def make_img_caption_pairs(self):
        for img_id in self.img_ids:
            caption_ids = self.coco_captions.getAnnIds(
                imgIds=int(img_id))

            captions = self.coco_captions.loadAnns(caption_ids)
            for caption in captions:
                caption = self._clean(caption['caption']) + ' <EOS>'
                self.img_caption_pairs.append((img_id, caption))

    def __len__(self):
        return len(self.img_caption_pairs)

    def __getitem__(self, index):
        img_id, caption = self.img_caption_pairs[index]
        return (
            np.load(FEATURES_DIR.format(img_id)),
            caption
        )

    def _clean(self, sentence):
        return sub(r'[^\w ]', '', sentence.lower()).strip()
