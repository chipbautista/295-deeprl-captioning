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
agent = Agent()
agent.actor.load_state_dict(torch.load(
    weights_file, map_location=None if USE_CUDA else 'cpu'
)['model_state_dict'])

MSCOCO_dataset = MSCOCO(split, evaluation=True)
loader = DataLoader(MSCOCO_dataset, shuffle=True)

img_ids = []
results = []
for i, (img_id, img_features, captions) in enumerate(loader):
    # for some unknown reason,
    # individual caption is a tuple with one element. need to do [0].
    # captions = [c[0] for c in captions]

    _, init_state, init_lstm_states = env.reset(img_features)
    with torch.no_grad():
        # sampled_caption, _ = agent.inference(
        #     init_state, init_lstm_states, env)
        greedy_caption, _ = agent.inference(
            init_state, init_lstm_states, env, 'greedy')

    # print('Ex: GT Caption: ', captions[2])
    # print('Sampled caption: ', sampled_caption,
    #       env.get_metrics(captions, sampled_words))
    # print('Greedy caption: ', greedy_caption,
    #       env.get_metrics(captions, greedy_words))

    results.append({
        'image_id': int(img_id[0]),
        'caption': greedy_caption
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
