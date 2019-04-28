
import json
from collections import Counter
from re import sub

import numpy as np

from settings import CAPTIONS_DIR, VOCABULARY_SIZE


with open(CAPTIONS_DIR.format('train')) as f:
    captions = [sub(r'[^\w ]', '', c['caption'].lower()).split()
                for c in json.loads(f.read())['annotations']]

counter = Counter(captions[0])
for c in captions[1:]:
    counter.update(c)

with open(CAPTIONS_DIR.format('val')) as f:
    captions = [sub(r'[^\w ]', '', c['caption'].lower()).split()
                for c in json.loads(f.read())['annotations']]

for c in captions:
    counter.update(c)

vocabulary = ['<SOS>', '<EOS>']

vocabulary.extend([w[0] for w in counter.most_common(VOCABULARY_SIZE)])
np.save('vocabulary', vocabulary)

print('Saved top {} words to vocabulary.npy'.format(VOCABULARY_SIZE))
