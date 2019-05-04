# Files
CAPTIONS_DIR = '../data/coco/annotations/captions_{}2014.json'
KARPATHY_SPLIT_DIR = '../data/karpathy_splits/coco2014_cocoid.{}.txt'
FEATURES_DIR = '../data/features/extracts/{}.npy'
MEAN_VEC_DIR = '../data/mean_vectors/{}.npy'
MODEL_DIR = '../models/{}'

# Neural Network Training Settings
LEARNING_RATE = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = 128
SHUFFLE = True
MAX_EPOCH = 50

# RL Training Settings
DISCOUNT_FACTOR = 0.95
MAX_WORDS = 30

# Data Settings
PAIR_ALL_CAPTIONS = True

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
