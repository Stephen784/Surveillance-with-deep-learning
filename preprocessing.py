from typing import List, Union

import numpy as np
import glob
import os
import math
import cv2
from numpy import ndarray
from tqdm import tqdm
import json


def frame_extractor(v_file, path='./', ext='.avi', frames_dir='train_1', extract_rate='all', frames_ext='.jpg'):

    os.makedirs(frames_dir, exist_ok=True)
    # capturing the video from the given path
    if ext not in v_file:
        v_file += ext
    cap = cv2.VideoCapture(path + v_file)

    frameRate = cap.get(5)
    os.makedirs(frames_dir + '/' + v_file, exist_ok=True)
    count = 0
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret:
            if type(extract_rate) == int:
                if extract_rate > frameRate:
                    print('Frame rate of Given Video: {0} fps'.format(frameRate))
                    raise ValueError(
                        'The value of `extract_rate` argument can not be greater than the Frame Rate of the video.')

                if (frameId % extract_rate == 0) and extract_rate > 1:

                    filename = frames_dir + '/' + v_file + '/' + "_frame{0}".format(count) + frames_ext
                    count += 1
                    cv2.imwrite(filename, frame)
                elif extract_rate == 1:
                    if frameId % math.floor(frameRate) == 0:
                        filename = frames_dir + '/' + v_file + '/' + "_frame{0}".format(count) + frames_ext
                        count += 1
                        cv2.imwrite(filename, frame)
            elif type(extract_rate) == int:
                if extract_rate == 'all':

                    filename = frames_dir + '/' + v_file + '/' + v_file + "_frame{0}".format(count) + frames_ext
                    count += 1
                    cv2.imwrite(filename, frame)
                else:
                    raise ValueError(
                        'Invalid Value for argument `extract_rate`, it can be either `all` or an integer value.')
            continue
        else:
            pass

        break
    cap.release()


def ReadFileNames(path, frames_ext='.tif'):

    directories: List[str] = [name for name in os.listdir(path)if os.path.isdir(os.path.join(path +'/'+ name))]
    onlyfiles = [0]
    file_names = [0]

    for i in range(len(directories)):
        files = glob.glob(path + '/' + directories[i] + '/*{0}'.format(frames_ext))
        names = []
        for file in files:
            names.append(file.split("\\")[1])
        file_names.append(names)
        onlyfiles.append(files)
    return onlyfiles, file_names, directories


def ToJson(obj, name, path='./', json_dir=False):

    if json_dir:
        os.makedirs(path + '/JSON', exist_ok=True)
        with open(path + '/JSON/{0}.json'.format(name), 'w') as f:
            json.dump(obj, f)
        f.close()
    elif not json_dir:
        if '.json' in name:
            pass
        else:
            name = name + '.json'
        with open(path + '/' + name, 'w') as f:
            json.dump(obj, f)
        f.close()


def ProcessImg(img_name, read_path, write=True, write_path=None, res_shape=(128, 128)):
    if write and write_path is None:
        raise TypeError(
            'The value of argument cannot be `None` when, `write` is set to True. Provide a valid path, where processed image should be stored!')
    img = cv2.imread(read_path)

    img = cv2.resize(img, res_shape)

    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(img, rgb_weights)

    if write:
        os.makedirs(write_path, exist_ok=True)
        cv2.imwrite(write_path + '/' + img_name, gray)
    return gray


def GlobalNormalization(img_list, name=None, path='Train_Data', save_data=True):
    img_arr: ndarray = np.array(img_list)
    batch, height, width = img_arr.shape

    img_arr.resize(height, width)

    img_arr = (img_arr - img_arr.mean()) / (img_arr.std())

    img_arr = np.clip(img_arr, 0, 1)
    if save_data:
        if name is None:
            raise TypeError(
                'The value of the `name` argument cannot be `None` type, when `save_data` is set to True. Provide value with `str` datatype.')
        if '.npy' not in name:
            name += '.npy'
        os.makedirs(path, exist_ok=True)
        np.save(path + '/' + name, img_arr)
        print('\n------ Data Save Successfully at this path: {0} ------\n'.format(path))
    return img_arr


def Vid2Frame(vid_path, frames_dir, ext_vid='.avi', frames_ext='.tif'):
    videos = glob.glob(vid_path + '/*{0}'.format(ext_vid))

    for vid in tqdm(videos):
        path = vid.split('\\')[0] + '/'
        v_file = vid.split('\\')[1]
        frame_extractor(v_file, path=path, ext=ext_vid, frames_dir=frames_dir,
                        extract_rate='all', frames_ext=frames_ext)


def Fit_Preprocessing(path: object, frames_ext: object) -> object:

    if frames_ext is None:
        raise TypeError(
            'Invalid Value for argument `frames_ext`, it cannot be None. Give proper extensions of the frames e.g: `.tif` or `.png` etc!')
    print('\n\nProcessing Images in this Dataset Path: {0}\n'.format(path))
    file_names: List[Union[str, List[str]]]
    onlyfiles, file_names, dirs = ReadFileNames(path, frames_ext)
    img_list = [1, 2, 3, 4]
    for img in tqdm(range(len(onlyfiles))):
        images = onlyfiles[img]
        count = 0
        for img.tif in {images}:
            img.split('/')
            img_name = dirs[i] + '_' + file_names[i][count]
            write_path = 'ProcessedImages/' + path.split('/')[1]
            gray = ProcessImg(img_name, read_path=img, write=True,
                              write_path=write_path, res_shape=(227, 227))
            img_list.append(gray)
            count += 1
    return img_list


if __name__ == '__main__':

    paths=['C:/Users/Public/Downloads/Surveillance with deep learning/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test', 'C:/Users/Public/Downloads/Surveillance with deearning/Datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2Test',
        'C:/Users/Public/Downloads/Surveillance with deep learning/Datasets/Avenue Dataset/testing_videos']
    path: str
    for path in paths:
        img_list: object = Fit_Preprocessing(path, frames_ext='.Fit')
        name: str = 'Test_{0}.npy'.format(path.split('/')[1])
        img_arr = GlobalNormalization(img_list, path='Test_Data', save_data=True)
