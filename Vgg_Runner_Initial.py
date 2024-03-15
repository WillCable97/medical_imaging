from src.data.DataLoaders.tf_record_loader import get_all_files #loader
from src.data.DataObjs.StdSingleImageSet import StdSingleImageSet
from src.models.VggModel.Vgg16 import Vgg16
import sys
import os 
import keras


root_dir = "./"
root_dir = os.path.abspath(root_dir)
data_path = os.path.join(root_dir, "data")
processed_path = os.path.join(data_path, "processed")

data_set = StdSingleImageSet(get_all_files, input_dir = processed_path)

final_data = data_set.batch_and_shuffle(20, 1000)
print(final_data)


model = Vgg16(4)


model.compile("adam", keras.losses.SparseCategoricalCrossentropy())
model.fit(final_data, epochs=10)