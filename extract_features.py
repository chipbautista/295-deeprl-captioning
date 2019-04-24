"""
Based on
https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/read_tsv.py
"""
import base64
import csv
import sys

import numpy as np

# need this or else csv won't read the file
# due to max exceeded something
csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h',
              'num_boxes', 'boxes', 'features']

with open('../data/features/trainval_36.tsv' 'r+b') as f:
    csv_reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)
    for item in csv_reader:
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])

        # do we need the boxes and the dimensions, though?
        for field in ['boxes', 'features']:
            item[field] = np.frombuffer(
                base64.decodestring(item[field]),
                dtype=np.float32).reshape((item['num_boxes'], -1))

        np.save('../data/features/{}.npy'.format(item['image_id']),
                item)
