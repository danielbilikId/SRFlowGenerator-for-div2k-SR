import torch

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

ALPHA = 0.1
RESIZE_HEIGHT = 256
RESIZE_WIDTH = 256
RATIO = 4
TRAIN_VAL_SPLIT_PERC = 0.8
VAL_TEST_SPLIT_PERC = 0.5
DATA_ROOT = './data'
TRAIN_HR_DIR = f'{DATA_ROOT}/DIV2K_train_HR'
VALID_HR_DIR = f'{DATA_ROOT}/DIV2K_valid_HR'
TENSOR_X_PATH = './preprocessed_data/tensor_x.npy'
TENSOR_Y_PATH = './preprocessed_data/tensor_y.npy'
VAL_TENSOR_X_PATH = './preprocessed_data/val_tensor_x.npy'
VAL_TENSOR_Y_PATH = './preprocessed_data/val_tensor_y.npy'

UPSCALE_FACTOR = 4
CHANNELS = 3

SRFLOW_NF = 64 
SRFLOW_NB = 16
SRFLOW_GC = 32