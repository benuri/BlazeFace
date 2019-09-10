from pandas import read_csv
import cv2
import glob
import os
import numpy as np
import logging
import coloredlogs
import tensorflow as tf
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)


IM_EXTENSIONS = ['png', 'jpg', 'jpeg', 'bmp']


def read_img(img_path, img_shape=(128, 128)):
    """
    load image file and divide by 255.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_shape)
    img = img.astype('float')
    img /= 255.

    return img


def append_zero(arr):
    return np.append([0], arr)



def dataloader(dataset_dir, label_path,  batch_size=1000, img_shape=(128, 128)):
    """
    data loader

    return image, [class_label, class_and_location_label]
    """

    label_df = read_csv(label_path)
    label_idx = label_df.set_index('filename')
    img_files = label_idx.index.unique().values

    numofData = len(img_files)  # endwiths(png,jpg ...)
    data_idx = np.arange(numofData)

    while True:

        batch_idx = np.random.choice(data_idx, size=batch_size, replace=False)

        batch_img = []
        batch_label = []
        batch_class = []

        for i in batch_idx:

            img = read_img(dataset_dir + img_files[i], img_shape=img_shape)

            label = label_idx.loc[img_files[i]].values
            label = np.array(label, ndmin=2)
            label = label[:, :4]

            cls_loc_label = np.apply_along_axis(append_zero, 1, label)

            batch_img.append(img)
            batch_label.append(cls_loc_label)   # face + bb
            # print(cls_loc_label[:, 0:1].shape)
            batch_class.append(cls_loc_label[:, 0:1])  # label[:, 0:1]) ---> face

            # yield {'input_1': np.array(batch_img, dtype=np.float32)}, {'clf_output': np.array(batch_class, dtype=np.float32),'bb_output': np.array(batch_label, dtype=np.float32)}
            
        yield np.array(batch_img, dtype=np.float32), [np.array(batch_class, dtype=np.float32), np.array(batch_label, dtype=np.float32)]

if __name__ == "__main__":
    pass
