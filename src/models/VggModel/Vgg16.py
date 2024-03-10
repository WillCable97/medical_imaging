import keras
from src.models.VggModel.ConvComponent import ConvComponent
from src.models.VggModel.DenseOutput import DenseOutput

class Vgg16(keras.Model):
    def __init__(self, output_classes:int):
        super().__init__()
        self.initial_pool = keras.layers.MaxPooling2D((5, 5), strides=(5, 5))
        self.comp_1 = ConvComponent(64)
        self.comp_2 = ConvComponent(128)
        self.comp_3 = ConvComponent(256) #all these have only 2 components (slight deviation)
        self.comp_4 = ConvComponent(512)
        self.comp_5 = ConvComponent(512)
        self.connected_layer = DenseOutput(output_classes)

    def call(self, inputs):
        inputs = self.initial_pool(inputs)
        inputs = self.comp_1(inputs)
        inputs = self.comp_2(inputs)
        inputs = self.comp_3(inputs)
        inputs = self.comp_4(inputs)
        inputs = self.comp_5(inputs)
        inputs = self.connected_layer(inputs)
        return inputs