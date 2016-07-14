
from __future__ import print_function
import os

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data import load_train_data, load_test_data
from train import dice_coef,dice_coef_loss

from skimage.transform import rotate, resize
from skimage import data

rows=160
cols=224

def preprocess(imgs, img_rows,img_cols):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def augmentation(image, imageB, org_width=160,org_height=224, width=190, height=262):
    max_angle=20
    image=resize(image,(width,height))
    imageB=resize(imageB,(width,height))

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
        imageB=cv2.flip(imageB,0)
    # image=resize(image,(org_width,org_height))

    return image,imageB
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    
# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f

def _up_block(block,mrge, nb_filters):
    up = merge([Convolution2D(2*nb_filters, 2, 2, border_mode='same')(UpSampling2D(size=(2, 2))(block)), mrge], mode='concat', concat_axis=1)
    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(up)
    conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(up)
    conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(conv)

    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 1, 1, activation='relu', border_mode='same')(conv)
    
    # conv = Convolution2D(4*nb_filters, 1, 1, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 3, 3, activation='relu', border_mode='same')(conv)
    # conv = Convolution2D(nb_filters, 1, 1, activation='relu', border_mode='same')(conv)

    return conv


# http://arxiv.org/pdf/1512.03385v1.pdf
# 50 Layer resnet
def resnet():
    input = Input(shape=(1, rows, cols))

    nb_filters=4 # 5
    conv1 = _conv_bn_relu(nb_filter=2*nb_filters, nb_row=7, nb_col=7, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = _residual_block(block_fn, nb_filters=2*nb_filters, repetations=3, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=2**2*nb_filters, repetations=4)(block1)
    block3 = _residual_block(block_fn, nb_filters=2**3*nb_filters, repetations=6)(block2)
    block4 = _residual_block(block_fn, nb_filters=2**4*nb_filters, repetations=3)(block3)

    up5=_up_block(block4,block3,2**3*nb_filters)
    up6=_up_block(up5,block2,2**2*nb_filters)
    up7=_up_block(up6,block1,2*nb_filters)
    up8=_up_block(up7,conv1,nb_filters)

    conv10=Convolution2D(1,1,1,activation='sigmoid')(up8)

    model = Model(input=input, output=conv10)
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def main():
    # import time
    # start = time.time()
    # model = resnet()
    # duration = time.time() - start
    # print("{} s to make model".format(duration))

    # start = time.time()
    # model.output
    # duration = time.time() - start
    # print("{} s to get output".format(duration))

    # start = time.time()
    # model.compile(loss="categorical_crossentropy", optimizer="sgd")
    # duration = time.time() - start
    # print("{} s to get compile".format(duration))

    # current_dir = os.path.dirname(os.path.realpath(__file__))
    # model_path = os.path.join(current_dir, "resnet_50.png")
    # plot(model, to_file=model_path, show_shapes=True)
    # exit()
# -----------------------------------------------------------------------------
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train, rows,cols)
    imgs_mask_train = preprocess(imgs_mask_train, rows/2,cols/2)

    imgs_train = imgs_train.astype('float32')
    mean = imgs_train.mean(0)[np.newaxis,:]  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = resnet()
    # model.load_weights('resnet.hdf5')
    
    model_checkpoint = ModelCheckpoint('resnet.hdf5', monitor='loss',verbose=1, save_best_only=True)
# ----------------------------------------------------------------------- 
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train, imgs_mask_train, batch_size=32, nb_epoch=20, verbose=1, shuffle=True, callbacks=[model_checkpoint])
    # for i in range(3):
    #     model.train(imgs_train[:3],imgs_mask_train[:3])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test, rows,cols)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('resnet.hdf5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)

    print('-'*30)
    print('Predicting masks on train data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_train, verbose=1)
    np.save('imgs_train_pred.npy', imgs_mask_test)

if __name__ == '__main__':
    main()