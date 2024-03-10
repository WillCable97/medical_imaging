import keras


class ConvComponent(keras.layers.Layer):
    def __init__(self, channel_count: int):
        self.conv_1 = keras.layers.Conv2D(channel_count, (3, 3), activation='relu', padding='same')
        self.conv_2 = keras.layers.Conv2D(channel_count, (3, 3), activation='relu', padding='same')
        self.pool_layer = keras.layers.MaxPooling2D((2,2), strides=(2,2))

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.pool_layer(x)
        return x


