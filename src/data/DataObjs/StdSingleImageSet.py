import tensorflow as tf

def parse_tfrecord_fn(example_proto):
    feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(example_proto, feature_description)
    
    height = tf.cast(example['height'], tf.int32)
    width = tf.cast(example['width'], tf.int32)
    depth = tf.cast(example['depth'], tf.int32)
    label = tf.cast(example['label'], tf.int32)

    image_raw = tf.io.decode_raw(example['image_raw'], tf.int32)
    image_raw = tf.cast(image_raw, tf.float32)


    image = tf.reshape(image_raw, (256, 256, 24)) #THIS NEED TO BE FIXED BUT IS REALLY ANNOYING
    label = tf.reshape(label, (1,))
    return image, label

class StdSingleImageSet:
    def __init__(self, input_func, **kwargs):
        image_data = input_func(**kwargs) #May need conversion, this is an np array
        tf_set = tf.data.TFRecordDataset(image_data)
        self.tf_set = tf_set.map(parse_tfrecord_fn)
        
    def batch_and_shuffle(self, batch_size: int, buffer_size: int):
        self.final_data =self.tf_set.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return self.final_data