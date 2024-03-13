import os 
import pandas as pd
import tensorflow as tf
import numpy as np
import pydicom as dicom
import cv2
import pickle


root_dir = "./"
root_dir = os.path.abspath(root_dir)
data_path = os.path.join(root_dir, "data")
raw_data_path = os.path.join(data_path, "raw")
processed_path = os.path.join(data_path, "interim")

#Label file
label_file = os.path.join(raw_data_path, "labels.csv")
d_frame = pd.read_csv(label_file)
d_frame["label"] = (d_frame["Normal"] + (2* d_frame["Cancer"]) 
                    + (3* d_frame["Actionable"]) + (4* d_frame["Benign"])) - 1


def get_label_info(file_path: str) -> pd.DataFrame:
    d_frame = pd.read_csv(file_path)
    d_frame["label"] = (d_frame["Normal"] + (2* d_frame["Cancer"]) 
                        + (3* d_frame["Actionable"]) + (4* d_frame["Benign"])) - 1
    
    ret_df = d_frame[["PatientID", "StudyUID", "label"]]
    ret_df = ret_df.drop_duplicates()

    return ret_df


label_info = get_label_info(label_file)


#fEATURE DEFINITIONS
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def extract_file_meta_data(raw_path: str, file_name: str):
    dframe = pd.read_csv(os.path.join(raw_path, file_name))
    meta_data = dframe[["Subject ID", "Study Date", "File Location"]]
    meta_data.rename(columns={"Subject ID" :"PatientID"}, inplace=True)   

    meta_data = meta_data.copy()
    meta_data.loc[:, 'file_path'] = meta_data.apply(lambda row: os.path.join(raw_path, os.path.normpath(row['File Location']), "1-1.dcm"), axis=1)
    
    return meta_data#[["Study Date", "PatientID", "file_path"]]
    


all_file_paths = extract_file_meta_data(raw_path=os.path.join(raw_data_path, "manifest-1617905855234")
                                          , file_name="metadata.csv")


def append_labe_data(input_df, label_df):
    print(input_df.shape)
    print(label_df.drop_duplicates().shape)

    input_df = input_df.reset_index()
    label_df = label_df.reset_index()

    df_result = pd.merge(input_df, label_df, on = "PatientID", how='left')

    df_result['StudyUID'] = df_result['StudyUID'].str.lower()
    df_result['File Location'] = df_result['File Location'].str.lower()
    df_result = df_result[df_result.apply(lambda row: row['StudyUID'].lower() in row['File Location'].lower() 
                                          if pd.notna(row['StudyUID']) and pd.notna(row['File Location']) else False, axis=1)]
    
    ret_df = df_result[["PatientID", "StudyUID", "label", "file_path"]]

    return ret_df


test = append_labe_data(all_file_paths, label_info)
print(test.shape)

def single_image_loader(input_path: str) -> np.array:
    output_height = 256
    output_width = 256
    ds = dicom.dcmread(input_path)
    ds[(0x0028, 0x0101)].value = 16
    pix_arr = ds.pixel_array
    pix_arr = [cv2.resize(slice_data, (output_width, output_height)) for slice_data in pix_arr]    
    np_img = np.array(pix_arr)
    depth = np_img.shape[0]
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img, depth


min_depth =100

for index, row in test.iterrows():
    path = row["file_path"]
    label = row["label"]
    save_list = []

    image_data, depth = single_image_loader(path)

    if depth < min_depth: min_depth=depth

    save_list.append(label)
    save_list.append(image_data.tolist())

    file_path = os.path.join(processed_path, f"record{index}.pkl")

    with open(file_path, 'wb') as file:
       pickle.dump(save_list, file)

    print(f"Min Depth: {min_depth}")