import os

START = '<sos>'
END = '<eos>'
NL_EMBEDDING_SIZE = 64
CODE_EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.6
NUM_LAYERS = 2
LR = 0.001
BATCH_SIZE = 100
MAX_EPOCHS = 100
PATIENCE = 10
VOCAB_CUTOFF_PCT = 5
LENGTH_CUTOFF_PCT = 95
MAX_VOCAB_EXTENSION = 50
BEAM_SIZE = 20
MAX_VOCAB_SIZE = 10000
FEATURE_DIMENSION = 128
NUM_CLASSES = 2

GNN_HIDDEN_SIZE = 64
GNN_LAYER_TIMESTEPS = 8
GNN_DROPOUT_RATE = 0.0
SRC_EMBEDDING_SIZE = 8
NODE_EMBEDDING_SIZE = 64

MODEL_LAMBDA = 0.5
LIKELIHOOD_LAMBDA = 0.3
OLD_METEOR_LAMBDA = 0.2
GEN_MODEL_LAMBDA = 0.5
GEN_OLD_BLEU_LAMBDA = 0.5
DECODER_HIDDEN_SIZE = 128
MULTI_HEADS = 4
NUM_TRANSFORMER_LAYERS = 2

# Download data from here: https://drive.google.com/drive/folders/1heqEQGZHgO6gZzCjuQD1EyYertN4SAYZ?usp=sharing
# DATA_PATH should point to the location in which the above data is saved locally

DATA_PATH = '/content/drive/MyDrive/public-inconsistency-detection-data'
#DATA_PATH = 'public-inconsistency-detection-data'

RESOURCES_PATH = os.path.join(DATA_PATH, 'resources')

# Download model resources from here: https://drive.google.com/drive/folders/1cutxr4rMDkT1g2BbmCAR2wqKTxeFH11K?usp=sharing
# MODEL_RESOURCES_PATH should point to the location in which the above resources are saved locally.

MODEL_RESOURCES_PATH = '/content/drive/MyDrive/inconsistency-detection-model-resources'
#MODEL_RESOURCES_PATH = 'inconsistency-detection-model-resources'
NL_EMBEDDING_PATH = os.path.join(MODEL_RESOURCES_PATH, 'nl_embeddings.json')
CODE_EMBEDDING_PATH = os.path.join(MODEL_RESOURCES_PATH, 'code_embeddings.json')
FULL_GENERATION_MODEL_PATH = os.path.join(MODEL_RESOURCES_PATH,
                                          'generation-model.pkl.gz')

#AST_DATA_PATH = 'public-inconsistency-detection-data'
AST_DATA_PATH = '/content/drive/MyDrive/jit_asts'

# Should point to where the output is to be saved
PREDICTION_DIR = 'out_update'
#PREDICTION_DIR = 'update_out'
DETECTION_DIR = 'out_detect'
#DETECTION_DIR = 'detect_out'

