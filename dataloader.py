import numpy as np
import cv2
import pydicom
import torch
from torch.utils import data as torch_data

def load_dicom_image3d(filelist_str, slice_num, img_size):
    
    dicom_files = filelist_str.split(';')[:-1]
    dicoms_all = []
    for d in dicom_files:
        try:
            dicom_data = pydicom.dcmread(d)
            if hasattr(dicom_data, 'pixel_array') and hasattr(dicom_data, 'ImagePositionPatient'):
                dicoms_all.append(dicom_data)
        except Exception as e:
            print(f"Error reading DICOM file {d}: {str(e)}")

    first_date = None
    first_time = None

    for dcm in dicoms_all:
        study_date = dcm.StudyDate
        study_time = dcm.StudyTime

        if first_date is None or (study_date < first_date) or (study_date == first_date and study_time < first_time):
            first_date = study_date
            first_time = study_time

    dicoms = []
    for dcm in dicoms_all:
        study_date = dcm.StudyDate
        study_time = dcm.StudyTime

        if study_date == first_date and study_time == first_time:
            dicoms.append(dcm)
    
    z_pos = [float(d.ImagePositionPatient[-1]) for d in dicoms]
    img_list = [cv2.resize(d.pixel_array, (img_size, img_size)) for d in dicoms]
    img_shape = img_list[0].shape
    img_list = [cv2.resize(img, (img_size, img_size)) for img in img_list if img.shape == img_shape]
    img_list = [img for _, img in sorted(zip(z_pos, img_list), key=lambda x: x[0])]
    img = np.stack(img_list)
    
    middle = len(dicoms)//2
    num_imgs2 = slice_num//2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(dicoms), middle + num_imgs2)
    img = img[p1:p2]

    if len(img) < slice_num:
        img = np.pad(img, ((0, slice_num - len(img)), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    # convert to HU
    M = float(dicoms[0].RescaleSlope)
    B = float(dicoms[0].RescaleIntercept)
    img = img * M + B
    
    # Windowing
    img = windowing(img)
    # cropping and padding
    img = np.stack([add_pad(crop_image(_)) for _ in img])
    
    return np.expand_dims(img, 0)

def windowing(img):
    X = np.clip(img.copy(), 15, 100)
    if np.min(X) < np.max(X):
        X = X - np.min(X)
        X = X / np.max(X)
    return X


def crop_image(image, display=False):

    mask = image == 0
    coords = np.array(np.nonzero(~mask))
    if coords.size > 0:
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        cropped_image = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
        return cropped_image
    else:
        return image

def add_pad(image, new_height=256, new_width=256):

    height, width = image.shape
    add_pad_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    add_pad_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return add_pad_image

class Dataset(torch_data.Dataset):
    def __init__(self, csv, pid, slice_num, img_size):
        self.csv = csv
        self.pid = pid
        self.slice_num = slice_num
        self.img_size = img_size
    
    def __len__(self):
        return len(self.pid)
    
    def __getitem__(self, index):
        id = self.pid[index]
        filelist_str = self.csv.loc[self.csv["pid"] == id, "headCT_path_list"].values[0]
        img = load_dicom_image3d(filelist_str, self.slice_num, self.img_size)
        
        y = self.csv.loc[self.csv["pid"] == id, "obj"].values[0]
        return {"X" : torch.tensor(img).float(), "y" : torch.tensor(y).float()}
