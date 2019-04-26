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

tsv_dir = '../data/features/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'

print('This will extract all the features for 120K images to ../data/feature/extracts. Will take a while.')
with open(tsv_dir, 'rb') as f:
    csv_reader = csv.DictReader(f, delimiter='\t', fieldnames=FIELDNAMES)
    for i, item in enumerate(csv_reader):
        features = np.frombuffer(
            base64.decodestring(item['features']),
            dtype=np.float32)

        # this results in shape (36, 2048), or 73728 values...
        # each one is 96 bytes, according to sys.getsizeof()
        features = features.reshape((int(item['num_boxes']), -1))

        np.save('../data/features/extracts/{}.npy'.format(item['image_id']),
                features)

        if i % 1000 == 0:
            print('Extracted features for {} images.'.format(i))

print('Extraction done.')