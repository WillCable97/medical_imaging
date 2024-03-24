import pydicom as dicom
import numpy as np

def single_image_loader(input_path: str) -> np.array:
    ds = dicom.dcmread(input_path)
    np_img = np.array(ds.pixel_array)
    np_img = np.transpose(np_img, (1, 2, 0))
    return np_img 

