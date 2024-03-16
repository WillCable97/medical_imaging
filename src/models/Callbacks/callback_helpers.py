import os 


def path_to_model_saves(base_path, input_model_name):
    path_to_model_saves = os.path.join(base_path, f"models/{input_model_name}")
    if not os.path.exists(path_to_model_saves): os.makedirs(path_to_model_saves)
    return path_to_model_saves


