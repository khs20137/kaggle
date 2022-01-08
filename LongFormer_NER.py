import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import *

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

VER = 12

LOAD_TOKENS_FROM = '../input/tf-longformer-v12'
LOAD_MODEL_FROM = '../input/long-v14'
DOWNLOADED_MODEL_PATH = '../input/tf-longformer-v12'

print(DOWNLOADED_MODEL_PATH)

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'allenai/longformer-base-4096'


if DOWNLOADED_MODEL_PATH == 'model':
    from transformers import AutoTokenizer, AutoConfig, TFAutoModel

    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained('model')
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.save_pretrained('model')

    backbone = TFAutoModel.from_pretrained(MODEL_NAME, config=config)
    backbone.save_pretrained('model')

print('TF version', tf.__version__)

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
print('Mixed precision enabled')

train = pd.read_csv('train.csv')
print(train.shape)
print(train.head())

assert (np.sum(train.groupby('id')['discourse_start'].diff() <= 0) == 0)

print('The train labels are:')
print(train.discourse_type.unique())

# The train labels are:
# ['Lead' 'Position' 'Evidence' 'Claim' 'Concluding Statement'
#  'Counterclaim' 'Rebuttal']

IDS = train.id.unique()
print('There are', len(IDS), 'train texts')

Max_LEN = 1024

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)


