from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data import load_train_data, load_test_data

from skimage.transform import rotate, resize
from skimage import data
import matplotlib.pyplot as plt
img_rows = 160
img_cols = 224

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def augmentation(image, imageB, org_width=160,org_height=224, width=190, height=262):
    max_angle=20
    image=cv2.resize(image,(height,width))
    imageB=cv2.resize(imageB,(height,width))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle
    image=rotate(image,angle,resize=True)
    imageB=rotate(imageB,angle,resize=True)

    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)
    image=image[xstart:xstart+org_width,ystart:ystart+org_height]
    imageB=imageB[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
        imageB=cv2.flip(imageB,1)
    
    if np.random.randint(2):
        image=cv2.flip(image,0)
        imageB=cv2.flip(imageB,0)

    image=cv2.resize(image,(org_height,org_width))
    imageB=cv2.resize(imageB,(org_height,org_width))

    return image,imageB
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)
    # pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool5)
    # convdeep = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(convdeep)
    
    # upmid = merge([Convolution2D(512, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(convdeep)), conv5], mode='concat', concat_axis=1)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(upmid)
    # convmid = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(convmid)

    up6 = merge([Convolution2D(256, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv5)), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([Convolution2D(128, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv6)), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([Convolution2D(64, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv7)), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([Convolution2D(32, 2, 2,activation='relu', border_mode='same')(UpSampling2D(size=(2, 2))(conv8)), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.float)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # imgs_train, imgs_mask_train = load_train_data()
    imgs_train=np.load("/mnt/data1/yihuihe/mnc/data.npy")
    imgs_mask_train=np.load("/mnt/data1/yihuihe/mnc/mask.npy")
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')

    # imgs_train = preprocess(imgs_train)
    # imgs_mask_train = preprocess(imgs_mask_train)
    # print(np.histogram(imgs_train))
    # print(np.histogram(imgs_mask_train))

    total=imgs_train.shape[0]
    # imgs_train/=255.
    # mean = imgs_train.mean()# (0)[np.newaxis,:]  # mean for data centering
    # std = np.std(imgs_train)  # std for data normalization
    # imgs_train -= mean
    # imgs_train /= std

    # imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    
    # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)

    # print('-'*30)
    # print('Fitting model...')
    # print('-'*30)
    # model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True,callbacks=[model_checkpoint])
    
    # batch_size=32
    # max_iters=10000
    # for i in range(max_iters):
    #     data_batch=np.ndarray((batch_size,1,img_rows,img_cols))
    #     mask_batch=np.ndarray((batch_size,1,img_rows,img_cols))
        
    #     for img in range(batch_size):
    #         idx=np.random.randint(total)
    #         data_batch[img,0],mask_batch[img,0]=augmentation(imgs_train[idx],imgs_mask_train[idx])
    #         # plt.subplot(121)
    #         # plt.imshow(data_batch[img,0])
    #         # plt.subplot(122)
    #         # plt.imshow(mask_batch[img,0])
    #         # plt.show()
    #         data_batch-=mean
    #         data_batch/=std
    #         print(np.histogram(data_batch))
    #         print(np.histogram(mask_batch))

    #     model.train_on_batch(data_batch,mask_batch)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test) # TODO: bug

    imgs_test = imgs_test.astype('float32')
    imgs_test -= np.load('/mnt/data1/yihuihe/mnc/mean.npy')
    imgs_test /=np.load('/mnt/data1/yihuihe/mnc/std.npy')

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('unet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
