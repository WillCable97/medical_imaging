import keras

class DenseOutput(keras.layers.Layer):
    def __init__(self, output_classes:int):
        super().__init__()
        self.flatten_layer = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(4096, activation='relu')
        self.dense_2 = keras.layers.Dense(4096, activation='relu')
        self.output_layer = keras.layers.Dense(output_classes, activation='softmax')

    def call(self, x):
        x = self.flatten_layer(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.output_layer(x)
        return x