import os
import tensorflow as tf
import pickle
import numpy as np 


root_dir = "./"
root_dir = os.path.abspath(root_dir)
data_path = os.path.join(root_dir, "data")
interim_data_path = os.path.join(data_path, "interim")
processed_path = os.path.join(data_path, "processed")

MINDEPTH = 24


#fEATURE DEFINITIONS
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

for i, file_found in enumerate(os.listdir(interim_data_path)):
    if file_found == '.gitkeep': continue
    file_path = os.path.join(interim_data_path, file_found)

    with open(file_path, 'rb') as file:
        loaded_list = pickle.load(file)


    label = loaded_list[0]
    #img_data = loaded_list[1]
    img_data = np.array(loaded_list[1])
    depth = img_data.shape[2]
    skips = depth // MINDEPTH

    #Hacky
    img_data = img_data.transpose(2, 0, 1)
    img_data = img_data[::skips][:MINDEPTH]
    img_data = img_data.transpose(1, 2, 0)

    file_path = os.path.join(processed_path, f"record{i}.tfrecords")

    convert_to_record(img_data, label, file_path)