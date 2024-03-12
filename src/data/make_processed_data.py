import os 
import pandas as pd
import tensorflow as tf
import numpy as np
import pydicom as dicom
import cv2


data_path = "./data"
data_path = os.path.abspath(data_path)
file_path = os.path.join(data_path, "raw", "med_records")
processed_path = os.path.join(data_path, "processed")

def extract_full_files_paths(raw_path: str, file_name: str, processed_path: str):
    dframe = pd.read_csv(os.path.join(raw_path, file_name))
    path_list = dframe['File Location'].to_list()
    diagnostic_flag = dframe['3rd Party Analysis'].to_list()

    diagnostic_flag = [0 if x=="NO" else 1 for x in diagnostic_flag]
    path_list = [os.path.join(raw_path, x, "1-1.dcm") for x in path_list]

    return path_list, diagnostic_flag

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def single_image_loader(input_path: str) -> np.array:
    output_height = 256
    output_width = 256
    ds = dicom.dcmread(input_path)
    ds[(0x0028, 0x0101)].value = 16
    pix_arr = ds.pixel_array
    pix_arr = [cv2.resize(slice_data, (output_width, output_height)) for slice_data in pix_arr]    
    np_img = np.array(pix_arr)
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img 

def convert_to_record(image_data, label, file_name):
    rows, columns, depth = image_data.shape
    print(f"Writting")

    writer = tf.io.TFRecordWriter(file_name)
    image_raw = image_data.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(columns),
        'depth': _int64_feature(depth),
        'label': _int64_feature(int(label)),
        'image_raw': _bytes_feature(image_raw)}))
    
    writer.write(example.SerializeToString())


path_list, label = extract_full_files_paths(file_path, "metadata.csv", processed_path)


for i, path in enumerate(path_list):
    img_data = single_image_loader(path)
    convert_to_record(img_data, 1, os.path.join(processed_path, f"record{i}.tfrecords"))



"""
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

    image_raw = tf.io.decode_raw(example['image_raw'], tf.int16)
    image = tf.reshape(image_raw, (height, width, depth))

    return image, label


# Create a dataset from the TFRecord file
dataset = tf.data.TFRecordDataset([os.path.join(processed_path, "test.tfrecords")])

# Map the parse_tfrecord_fn function to each element in the dataset
dataset = dataset.map(parse_tfrecord_fn)

# Create an iterator and get the next element
iterator = iter(dataset)
image, label = iterator.get_next()


print(label)
print(img_data.reshape(-1).shape)
"""