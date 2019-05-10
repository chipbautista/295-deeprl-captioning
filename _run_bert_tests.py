import numpy as np
from sklearn.metrics import pairwise_distances

from pycocotools.coco import COCO
from bert_serving.client import BertClient

from settings import KARPATHY_SPLIT_DIR, CAPTIONS_DIR


def dist_respective(captions, metric):
    distances = []
    for i in range(5):
        for k in range(i + 1, 5):
            distances.append(
                pairwise_distances([captions[i]], [captions[k]], metric)
            )
    distances = np.array(distances)
    return distances.mean(), distances.std()


def dot_respective(captions):
    distances = []
    for i in range(5):
        for k in range(i + 1, 5):
            distances.append(np.dot(captions[i], captions[k]))
    distances = np.array(distances)
    return distances.mean(), distances.std()


def hanxiao(captions):
    # one vs all
    scores = []
    for i in range(5):
        x = captions[i]
        y = np.concatenate((captions[0:i], captions[i + 1:]))
        scores.append(
            np.sum(x * y, axis=1) / np.linalg.norm(y, axis=1))
    scores = np.array(scores)
    return scores.mean(), scores.std()


def trial(context_vectors):
    eucl_records = np.array([
        dist_respective(context_vectors[i: i + 5], 'euclidean')
        for i in range(0, NUM_SAMPLES * 5, 5)
    ])
    print('Mean Eucl Distance: ', eucl_records[:, 0].mean(),
          'with STD: ', eucl_records[:, 0].std())
    print('Mean Eucl Distance STD: ', eucl_records[:, 1].mean(),
          'with STD: ', eucl_records[:, 1].std())

    manh_records = np.array([
        dist_respective(context_vectors[i: i + 5], 'manhattan')
        for i in range(0, NUM_SAMPLES * 5, 5)
    ])
    print('Mean Manhattan Distance: ', manh_records[:, 0].mean(),
          'with STD: ', manh_records[:, 0].std())
    print('Mean Manhattan Distance STD: ', manh_records[:, 1].mean(),
          'with STD: ', manh_records[:, 1].std())

    dot_records = np.array([
        hanxiao(context_vectors[i: i + 5])
        for i in range(0, NUM_SAMPLES * 5, 5)
    ])
    print('Mean Dot Distance: ', dot_records[:, 0].mean(),
          'with STD: ', dot_records[:, 0].std())
    print('Mean Dot Distance STD: ', dot_records[:, 1].mean(),
          'with STD: ', dot_records[:, 1].std())


NUM_SAMPLES = 500

bc = BertClient()

with open(KARPATHY_SPLIT_DIR.format('train')) as f:
    img_ids = f.read().split('\n')[:-1]


for t in range(3):
    print('----- Trial ', t + 1, '-----')
    np.random.shuffle(img_ids)  # inplace operation
    img_ids_ = list(map(int, img_ids))[:NUM_SAMPLES]
    coco = COCO(CAPTIONS_DIR.format('train'))

    captions = [c['caption']
                for c in coco.loadAnns(coco.getAnnIds(img_ids_))]
    print('Extracting BERT context vectors...')
    context_vectors = bc.encode(captions)
    trial(context_vectors)

    # Try with unpaired captions
    print('--- UNPAIRED ---')
    cv = np.random.permutation(context_vectors)
    trial(cv)
