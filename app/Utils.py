import pandas as pd
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH')
MODEL_PATH = os.getenv('MODEL_PATH')
FEATURES_LIST = os.getenv('FEATURES_LIST').split(',')


def load_data(path=None, exclude=False):
    csv = DATA_PATH if not path else path
    df = pd.read_csv(csv)
    if exclude:
        features_to_drop = [x for x in df.columns if x not in FEATURES_LIST]
        df.drop(features_to_drop, axis=1, inplace=True)
    return df


def loadModel():
    with open(MODEL_PATH, 'rb') as path:
        return pickle.load(path)
