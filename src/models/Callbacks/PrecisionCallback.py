import numpy as np
from sklearn.metrics import precision_score
from tensorflow.keras.callbacks import Callback

class PrecisionCallback(Callback):
    def __init__(self, validation_data):
        super(PrecisionCallback, self).__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        y_val_true = []
        y_val_pred = []
        
        for x_batch_val, y_batch_val in self.validation_data:
            y_val_true.extend(np.argmax(y_batch_val, axis=1))
            y_val_pred.extend(np.argmax(self.model.predict(x_batch_val), axis=1))
        
        precision_per_class = precision_score(y_val_true, y_val_pred, average=None)
        
        print("\nPrecision for each class:")
        for i, precision in enumerate(precision_per_class):
            print(f"Class {i}: {precision:.4f}")
