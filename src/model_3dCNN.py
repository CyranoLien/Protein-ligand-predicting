import pickle
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Activation, BatchNormalization, concatenate, Dense, Input, MaxPooling3D, Flatten
from keras.layers.convolutional import Conv3D
from keras.preprocessing import sequence

def model_3dcnn(proshape, ligshape):
    pro = Input(shape=proshape, name='pro')
    x1 = Conv3D(32, (5, 5, 5), strides=(2, 2, 2), padding='valid')(pro)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling3D(strides=(2, 2, 2))(x1)
    # (6,6,6)
    x1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='valid')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling3D(strides=(2, 2, 2))(x1)
    # (2,2,2)


    lig = Input(shape=ligshape, name='lig')
    x2 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='valid')(lig)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling3D(strides=(2, 2, 2))(x2)
    # (2,2,2)

    x = concatenate([x1, x2])
    #x = Concatenate(x1, x2)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)

    output = Dense(1, activation='tanh')(x)
    model = Model(inputs=[pro, lig], output=output)
    return model


if __name__ == '__main__':

    with open('../data/cnn_data/cnn_pro_train.bin', 'rb') as f:
        cnn_pro_train = np.array(pickle.load(f))
    with open('../data/cnn_data/cnn_lig_train.bin', 'rb') as f:
        cnn_lig_train = np.array(pickle.load(f))
    with open('../data/cnn_data/cnn_out_train.bin', 'rb') as f:
        cnn_out_train = np.array(pickle.load(f))

    with open('../data/cnn_data/cnn_pro_valid.bin', 'rb') as f:
        cnn_pro_valid = np.array(pickle.load(f))
    with open('../data/cnn_data/cnn_lig_valid.bin', 'rb') as f:
        cnn_lig_valid = np.array(pickle.load(f))
    with open('../data/cnn_data/cnn_out_valid.bin', 'rb') as f:
        cnn_out_valid = np.array(pickle.load(f))

    print(cnn_pro_train.shape)
    print(cnn_lig_train.shape)


    model = model_3dcnn([27,27,27,1],[6,6,6,1])
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    hist = model.fit(x=[cnn_pro_train, cnn_lig_train], y=cnn_out_train, epochs=1, verbose=1,
                     validation_data=([cnn_pro_valid, cnn_lig_valid], cnn_out_valid))