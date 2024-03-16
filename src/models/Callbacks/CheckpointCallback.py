import os 
from src.models.Callbacks.callback_helpers import path_to_model_saves
from keras.callbacks import ModelCheckpoint

def checkpoint_callback(base_path, input_model_name, period):
    full_path = path_to_model_saves(base_path, input_model_name)
    full_path = os.path.join(full_path, "checkpoint_tracker")
    if not os.path.exists(full_path): os.makedirs(full_path)
    checkpoint_prefix = os.path.join(full_path, "ckpt_{epoch}.weights.h5",)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_prefix
        ,save_weights_only=True
        ,save_freq='epoch')
    return checkpoint_callback