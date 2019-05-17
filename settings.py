from torch import cuda
USE_CUDA = cuda.is_available()
LOAD_IMAGES_TO_MEMORY = False

# Files
CAPTIONS_DIR = '../data/coco/annotations/captions_{}2014.json'
KARPATHY_SPLIT_DIR = '../data/karpathy_splits/karpathy_{}_images.txt'
FEATURES_DIR = '../data/features/extracts/{}.npy'
CAPTION_VECTORS_DIR = '../data/caption_vectors/train/{}.npy'
MODEL_DIR = '../models/{}'

# Neural Network Training Settings
LEARNING_RATE = 0.001
MOMENTUM = 0.9
LR_DECAY_PER_EPOCH = 0.9
LR_DECAY_STEP_SIZE = 1  # decay every epoch
BATCH_SIZE = 128
SHUFFLE = True
MAX_EPOCH = 50

# RL Training Settings
LEARNING_RATE_RL = 1e-4
BATCH_SIZE_RL = 32
MAX_WORDS = 30
MODEL_WEIGHTS = MODEL_DIR.format('RL-0516-1547-E9')
BETA = 1  # if 1: pure cider score
TARGET_DIST = 180.0  # manhattan distance
INCLUDE_CONTEXT_SCORE = True

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
