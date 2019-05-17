import json
import sys

# ye, doing this the CS freshie way yeseeeess
sys.path.append('pycocoevalcap')
sys.path.append('pycocoevalcap/bleu')
sys.path.append('pycocoevalcap/meteor')
sys.path.append('pycocoevalcap/rouge')
sys.path.append('pycocoevalcap/spice')
sys.path.append('pycocoevalcap/cider')

import torch
from torch.utils.data import DataLoader
from pycocoevalcap.eval import COCOEvalCap

from agent import Agent
from environment import Environment
from data import MSCOCO
from settings import *


# too lazy to implement args...
# Given a command "python my_script.py ABC DEF"
# sys.argv[1] corresponds to "ABC"
# and sys.argv[2] corresponds to 'DEF'
# Useful for specifying which split to evaluate on. (default is test.)
try:
    split = sys.argv[1]
except IndexError:
    split = 'test'

try:
    weights_file = sys.argv[2]
except IndexError:
    weights_file = MODEL_WEIGHTS

env = Environment()
agent = Agent(env=env)
agent.actor.load_state_dict(torch.load(
    weights_file, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])

MSCOCO_dataset = MSCOCO(split, evaluation=True)
data_loader = DataLoader(MSCOCO_dataset, batch_size=500, shuffle=True)

results = []
for img_ids, img_features, captions in data_loader:
    with torch.no_grad():
        greedy_captions = agent.predict_captions(
            img_features, 'greedy', constrain=True)
    for img_id, greedy_caption in zip(img_ids, greedy_captions):
        results.append({
            'image_id': int(img_id),
            'caption': greedy_caption[0]
        })

with open(RESULT_DIR.format(split), 'w+') as f:
    f.write(json.dumps(results))

print('Saved results to COCO format in ', RESULT_DIR.format(split))
print('Will now run evaluation tool.')

# Following:
# https://github.com/tylin/coco-caption/blob/master/cocoEvalCapDemo.ipynb
coco = MSCOCO_dataset.coco  # already instantiated before
cocoRes = coco.loadRes(RESULT_DIR.format(split))

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
cocoEval.evaluate()

# Optional: code to see low-scoring captions. See their ipynb for usage.
