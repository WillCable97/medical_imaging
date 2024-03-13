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


import sys
import os 

def get_all_files(input_dir: str):
    print(input_dir)
    file_list =[os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    file_list = file_list[1:]#.gitkeep
    return file_list



root_dir = "./"
root_dir = os.path.abspath(root_dir)
data_path = os.path.join(root_dir, "data")
processed_path = os.path.join(data_path, "processed")


#from src.data.DataLoaders.tf_record_loader import get_all_files
A = StdSingleImageSet(get_all_files, input_dir = processed_path)

final_data = A.batch_and_shuffle(20, 1000)
print(final_data)