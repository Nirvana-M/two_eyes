import cv2,os,glob
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from numpy.random import permutation
from keras.models import load_model

from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.models import Model, model_from_json
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.engine.training import Model as KerasModel

from keras.losses import binary_crossentropy
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

#from resnet import build_resnet_18 
import resnet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", help="Set training data set as car/mri/dog", type=str)
parser.add_argument("--train_percent", help="Set the percentage of data used for training", type=int, default=100)
parser.add_argument("--unet_epochs", help="Set number of epochs for unet", type=int, default=100)

parser.add_argument("--loss_option", help="Set the opition for loss function", type=int, default=0) 




args = parser.parse_args()

data_path = '../input/'+args.dataset+'_data/'
unet_ep = args.unet_epochs
train_no = int(800 / 100 * args.train_percent)
loss_option = args.loss_option



unet_h, unet_w, unet_c = 128, 128, 3
batch_size = 16
threshold = 0.5

df_train = pd.read_csv(data_path + 'train_masks.csv')
ids_all = df_train['img'].map(lambda s: s.split('.')[0])
ids_train = ids_all[:train_no]
ids_val = ids_all[800:900]
ids_test = ids_all[900:1000]

#ids_train1 = ids_all[:800]
#ids_train2 = ids_all[1000:]
#ids_train = np.concatenate((ids_train1, ids_train2),axis=-1)



def load_data_unet(ids_input):

    train_x, train_y, train_y2 = [], [], []
    for id in ids_input:#files[:100]:
        img = cv2.imread(data_path +'train_hq/{}.jpg'.format(id))
        msk = cv2.imread(data_path +'train_masks/{}_mask.png'.format(id),0)
        img = cv2.resize(img,(unet_w,unet_h))
        msk = cv2.resize(msk,(unet_w,unet_h))
        
        msk2 = 255 - msk
        msk = np.expand_dims(msk, axis = 2)
        msk2 = np.expand_dims(msk2, axis = 2) 
        
        train_x.append(img)
        train_y.append(msk)
        train_y2.append(msk2)
    train_x = np.array(train_x, np.float32) / 255
    train_y = np.array(train_y, np.float32) / 255
    train_y2 = np.array(train_y2, np.float32) / 255
    return train_x, train_y, train_y2



def union_loss(y_true, y_pred):
    #import pdb; pdb.set_trace()
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    full_mask = K.ones_like(y_pred_f)
    score = (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)/(K.sum(full_mask) + smooth)
    return 1 - score



def dice_coeff(y_true, y_pred):
    #import pdb; pdb.set_trace()
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def two_eye_loss(y2_pred):
    def loss(y_true, y_pred):
        loss1 = dice_loss(y_true,y_pred)
        loss2 = dice_coeff(y_pred,y2_pred)
        loss3 = union_loss(y_pred,y2_pred)                

        return loss1 + 0.5*(loss2+loss3)
    return loss



def two_eye_loss_v1(y1,y2):
    def loss(y_true, y_pred):
        loss10 = dice_loss(y1,y2)
        loss11 = union_loss(y1,y2)
        loss12 = loss10+loss11
        loss1 = y_pred-y_true
        loss2 = K.abs(y_pred-loss12)

        return loss1 + loss2
    return loss


def get_unet_2eye(input_shape=(unet_h, unet_w, unet_c),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    #down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    #down4 = BatchNormalization()(down4)
    #down4 = Activation('relu')(down4)
    #down4 = Conv2D(512, (3, 3), padding='same')(down4)
    #down4 = BatchNormalization()(down4)
    #down4 = Activation('relu')(down4)
    #down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(512, (3, 3), padding='same')(down3_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(512, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    #up4 = UpSampling2D((2, 2))(center)
    #up4 = concatenate([down4, up4], axis=3)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    #up4 = Conv2D(512, (3, 3), padding='same')(up4)
    #up4 = BatchNormalization()(up4)
    #up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(center)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify1 = Conv2D(num_classes, (1, 1), activation='sigmoid', name = 'out1')(up1)


    up32 = UpSampling2D((2, 2))(center)
    up32 = concatenate([down3, up32], axis=3)
    up32 = Conv2D(256, (3, 3), padding='same')(up32)
    up32 = BatchNormalization()(up32)
    up32 = Activation('relu')(up32)
    up32 = Conv2D(256, (3, 3), padding='same')(up32)
    up32 = BatchNormalization()(up32)
    up32 = Activation('relu')(up32)
    up32 = Conv2D(256, (3, 3), padding='same')(up32)
    up32 = BatchNormalization()(up32)
    up32 = Activation('relu')(up32)
    # 32

    up22 = UpSampling2D((2, 2))(up32)
    up22 = concatenate([down2, up22], axis=3)
    up22 = Conv2D(128, (3, 3), padding='same')(up22)
    up22 = BatchNormalization()(up22)
    up22 = Activation('relu')(up22)
    up22 = Conv2D(128, (3, 3), padding='same')(up22)
    up22 = BatchNormalization()(up22)
    up22 = Activation('relu')(up22)
    up22 = Conv2D(128, (3, 3), padding='same')(up22)
    up22 = BatchNormalization()(up22)
    up22 = Activation('relu')(up22)
    # 64

    up12 = UpSampling2D((2, 2))(up22)
    up12 = concatenate([down1, up12], axis=3)
    up12 = Conv2D(64, (3, 3), padding='same')(up12)
    up12 = BatchNormalization()(up12)
    up12 = Activation('relu')(up12)
    up12 = Conv2D(64, (3, 3), padding='same')(up12)
    up12 = BatchNormalization()(up12)
    up12 = Activation('relu')(up12)
    up12 = Conv2D(64, (3, 3), padding='same')(up12)
    up12 = BatchNormalization()(up12)
    up12 = Activation('relu')(up12)
    # 128

    classify2 = Conv2D(num_classes, (1, 1), activation='sigmoid', name = 'out2')(up12)


    comb = concatenate([inputs, classify1, classify2], axis=3)

    classify3 = resnet.ResnetBuilder.build_resnet_18(comb, 1)

    predictions  = Dense(1, activation='linear', name = 'out3')(classify3)



    if loss_option == 1: # two branches 
        
        model = Model(inputs=inputs, outputs=[classify1, classify2])

        model.compile(optimizer=Adam(lr=0.001), loss={'out1':two_eye_loss(classify2), 'out2':two_eye_loss(classify1)},\
            loss_weights={'out1':.5, 'out2':.5}, metrics={'out1':dice_coeff, 'out2':dice_coeff})

    elif loss_option == 2:
        
        model = Model(inputs=inputs, outputs=[classify1, classify2, predictions])

        model.compile(optimizer=Adam(lr=0.001), loss={'out1':bce_dice_loss, 'out2':bce_dice_loss, 'out3':two_eye_loss_v1(classify1,classify2)},\
            loss_weights={'out1':.45, 'out2':.45, 'out3': .1}, metrics={'out1':dice_coeff, 'out2':dice_coeff})


    else: #loss_option == 0: # standard unet

        model = Model(inputs=inputs, outputs=classify1)

        model.compile(optimizer=Adam(lr=0.001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model





def train_model_unet_2eye(epochs):

    model = get_unet_2eye()
    model.summary()
    #train_data,train_target = load_data_unet(ids_train)
    X_train, Y_train, Y2_train = load_data_unet(ids_train)
    X_val, Y_val, Y2_val = load_data_unet(ids_val)


    Y3_train = np.zeros(len(X_train),np.float32)
    Y3_val = np.zeros(len(X_val),np.float32)

    if loss_option == 0: 
        best_model_file = "./weights/weights_2eye.hdf5"
    elif loss_option == 1:
        best_model_file = "./weights/weights_2eye_1.hdf5"
    else:
        best_model_file = "./weights/weights_2eye_2.hdf5"
    
    best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose = 1, save_best_only = True, mode = 'min')

    # this is the augmentation configuration we will use for training
    datagen = ImageDataGenerator(
            #rescale=1./255,
            shear_range=0.0,
            zoom_range=0.1,
            rotation_range=0.0,
            width_shift_range=0.0625,
            height_shift_range=0.0625,
            vertical_flip = True,
            horizontal_flip = False)

    for epoch in range(epochs):
        print 'Training and Validation Samples: '
        print X_train.shape, Y_train.shape, Y2_train.shape, Y3_train.shape, X_val.shape, Y_val.shape, Y2_val.shape, Y3_val.shape
        print('epoch {} over total {}'.format(epoch, epochs))

        
        if loss_option == 1:
            train_input_y = [Y_train, Y2_train]
            val_input_y = [Y_val, Y2_val]
        elif loss_option == 2:
            train_input_y = [Y_train, Y2_train, Y3_train]
            val_input_y = [Y_val, Y2_val, Y3_val]
        else:
            train_input_y = Y_train
            val_input_y = Y_val

        # fits the model on batches with real-time data augmentation:
        # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=16),
        model.fit(X_train, train_input_y, batch_size = 32,
                    #samples_per_epoch=len(X_train),
                    nb_epoch=1,
                    verbose=1,
                    validation_data = (X_val, val_input_y),
                    #validation_data = datagen.flow(X_val, Y_val, batch_size=16),
                    #nb_val_samples = len(X_val),
                    callbacks = [best_model])
        vIoU, vDice, vAcc = unet_predict_2eye(model, ids_test)
        print("epoch {:d} prediction results:".format(epoch))
        print('mean IoU = {:.4f}, mean Dice = {:.4f}, mean Acc = {:.4f}'.format(np.mean(vIoU), np.mean(vDice), np.mean(vAcc)))
 



def unet_predict_2eye(model, ids_input):
    X, y, y2 = load_data_unet(ids_input)
    y11 = model.predict(X)
    if loss_option == 0:
        y1 = y11
    else:
        y1 = y11[0]
    arrIoU = []
    arrDice = []
    arrAcc = []

    for i in range(len(y1)):
        mask = y1[i] > threshold
        mask = mask * 1.0
        label = y[i]
        comb = mask + label
        L2 = np.sum(comb==2)
        L1 = np.sum(comb==1)
        IoU = L2*1.0 / (L1+L2)
        Dice = L2*2.0 / (L1+2*L2)
        arrIoU.append(IoU)
        arrDice.append(Dice)

        #acc
        L0 = np.sum(comb==0)
        Acc = (L0 + L2)*1.0 / (L0 + L1 + L2)
        arrAcc.append(Acc)

    arrIoU = np.array(arrIoU)
    arrDice = np.array(arrDice)
    #acc
    arrAcc = np.array(arrAcc)
    print('mean IoU = {:.4f}, mean Dice = {:.4f}, mean Acc = {:.4f}'.format(np.mean(arrIoU), np.mean(arrDice), np.mean(arrAcc)))
    return np.mean(arrIoU), np.mean(arrDice), np.mean(arrAcc)



train_model_unet_2eye(unet_ep)

