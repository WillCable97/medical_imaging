import os 
import keras
from src.models.Callbacks.callback_helpers import path_to_model_saves
from keras.callbacks import CSVLogger

def csv_callback(base_path, input_model_name):
    full_path = path_to_model_saves(base_path, input_model_name)
    full_path = os.path.join(full_path, "csv_tracker")
    if not os.path.exists(full_path): os.makedirs(full_path)
    csv_prefix = os.path.join(full_path, "csv_tracker")
    csv_callback = CSVLogger(csv_prefix)
    return csv_callback