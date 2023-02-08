from mushroom.exception import Mushroom_Exception
from mushroom.logger import logging
from mushroom.predictor import ModelResolver
from mushroom.utils import load_object

import pandas as pd
import os,sys
from datetime import datetime
import numpy as numpy

PREDICTION_DIR="prediction"

def initiate_batch_prediction(input_filr_path):
    try:
        os.makedirs(PREDICTION_DIR, exist_ok=True)

