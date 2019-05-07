from torch import cuda
USE_CUDA = cuda.is_available()

# Files
CAPTIONS_DIR = '../data/coco/annotations/captions_{}2014.json'
KARPATHY_SPLIT_DIR = '../data/karpathy_splits/coco2014_cocoid.{}.txt'
FEATURES_DIR = '../data/features/extracts/{}.npy'
MEAN_VEC_DIR = '../data/mean_vectors/{}.npy'
MODEL_DIR = '../models/{}'

# Neural Network Training Settings
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
LR_DECAY_PER_EPOCH = 0.9
BATCH_SIZE = 128
SHUFFLE = True
MAX_EPOCH = 50

# RL Training Settings
# I dont understand why it needs a very small LR to stabilize.
# Any higher than 5e-8 and you'll get -inf's.
LEARNING_RATE_RL = 5e-8
BATCH_SIZE_RL = 30
DISCOUNT_FACTOR = 0.95
MAX_WORDS = 30
MODEL_WEIGHTS = MODEL_DIR.format('0505-2050-E10')  # 0506-1246-E6

# Data Settings
PAIR_ALL_CAPTIONS = True
RESULT_DIR = '../data/{}_results.json'
#  http://cocodataset.org/#format-results


# Network Architecture
# Follows Bottom-Up Top-Down paper.
LSTM_HIDDEN_UNITS = 1000
ATTENTION_HIDDEN_UNITS = 512
WORD_EMBEDDING_SIZE = 1000
VOCABULARY_SIZE = 10000 + 2  # 10,000 most common words plus <SOS> and <EOS>

IMAGE_FEATURE_DIM = 2048
IMAGE_FEATURE_REGIONS = 36

ATTENTION_LSTM_INPUT_SIZE = (
    LSTM_HIDDEN_UNITS +
    IMAGE_FEATURE_DIM +
    WORD_EMBEDDING_SIZE
)

LANGUAGE_LSTM_INPUT_SIZE = (
    IMAGE_FEATURE_DIM +
    LSTM_HIDDEN_UNITS
)
