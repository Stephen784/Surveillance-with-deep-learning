var = 3

import sys

import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFont, ImageDraw, Image
from tensorflow.keras.models import Model



def BuildModel(input_shape=(227, 227, 10, 1)):
    global input
    if len(input_shape) != 4 or type(input_shape) != tuple:
        raise ValueError('Invalid value given to the argument `input_shape`, it must be a `tuple` containing 4 values in this manner: (height, width, frames_per_input, channels)')
    input = input()

    spatial_enc = Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tan')(input)
    spatial_enc = Conv3D(filters=64, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tan')(spatial_enc)


    temporal_enc = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4, recurrent_dropout=0.3, return_sequences=True)(spatial_enc)
    temporal_enc = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3, return_sequences=True)(temporal_enc)


    temporal_dec = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, return_sequences=True, padding='same', dropout=0.5)(temporal_enc)


    spatial_dec = Conv3DTranspose(filters=128, kernel_size=(5, 5, 1), strides=(2, 2, 1), padding='valid', activation='tan')(temporal_dec)
    spatial_dec = Conv3DTranspose(filters=1, kernel_size=(11, 11, 1), strides=(4, 4, 1), padding='valid', activation='tan')(spatial_dec)


    model = Model(inputs=input, outputs=spatial_dec)
    model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def TF_GPUsetup(GB=4):

    if type(GB)!=int:
        raise TypeError('Type of Parameter `GB` must be `int` and it should be 1 GB less than your GPU memory')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    config = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=(1024*GB))]
    if gpus:

      try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], config)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    print('\nTensorflow GPU installed: '+str(tf.test.is_built_with_cuda()))
    print('Is Tensorflow using GPU: '+str(tf.test.is_gpu_available()))
    
def PrepareData(X_train, re_shape=(-1, 227, 227, 10)):
    global x_train
    frames = X_train.shape[2]

    frames -= frames%10
    x_train = x_train[:, :, : frames]
    x_train = x_train.reshape(re_shape)
    x_train=np.expand_dims(x_train, axis=4)
    return x_train
    

def GetTrainData(name, re_shape=(-1, 227, 227, 10)):
    if type(name)!=str:
        raise TypeError('Provide a valid name of `string` datatype, to the `.npy` file.')
    if '.npy' not in name:
        name += '.npy'
    X_train = np.load(name)
    return PrepareData(X_train, re_shape)

def OverlayText2Img(img, text):

    cv2_im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)
    

    font = ImageFont.truetype("arial.ttf", 40)
    

    draw.text((0, 0), text, font=font)
    

    im_pros = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    return im_pros

def ShowVideo(cap, v_frame, text):

    v_frame = OverlayText2Img(v_frame, text)
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('Real Time Anomaly Detection - Github.com/danish11', v_frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        raise KeyboardInterrupt('Real Time Anomaly Detection Stopped due to Keyboard Interrupt!')
            
def PrintInline(string):
    sys.stdout.write('\r'+string)
    sys.stdout.flush() 

def ImgProcess(frame, shape=(227, 227)):
    frame = cv2.resize(frame, shape)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    gray = np.dot(frame, rgb_weights)
    gray = (gray-gray.mean())/gray.std()
    gray = np.clip(gray, 0, 1)
    return gray

def Img_LstArr(img_lst, re_shape = (227, 227, 10)):
    img_arr = np.array(img_lst)
    img_arr.resize(re_shape)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = np.expand_dims(img_arr, axis=4)
    return img_arr

def MSE(x1, x2):

    diff = x1-x2
    a, b, c, d, e = diff.shape
    n_samples = a*b*c*d*e
    sq_diff = diff**2
    Sum = sq_diff.sum()
    dist = np.sqrt(Sum)
    mean_dist = dist/n_samples
    
    return mean_dist

def ListCopy(lst):
    new_lst = []
    for item_lst in lst:
        for item in item_lst:
            #ten_lst = []
            new_lst.append(item)        
    return new_lst

