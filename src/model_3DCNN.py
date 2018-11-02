import pickle
import heapq
import numpy as np
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Activation, BatchNormalization, concatenate, Dense, Input, MaxPooling3D, Flatten
from keras.layers.convolutional import Conv3D
from preprocess_train import sample_neg
import matplotlib.pyplot as plt


NUM_NEG = 2
NUM_TRAIN = 30
NUM_VALID = 20
NUM_TEST = 10
EPOCH = 10

PATH_cnn_pro_train = '../data/cnn_data/cnn_pro_train.bin'
PATH_cnn_lig_train = '../data/cnn_data/cnn_lig_train.bin'
PATH_cnn_pro_test = '../data/cnn_data/cnn_pro_test.bin'
PATH_cnn_lig_test = '../data/cnn_data/cnn_lig_test.bin'


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
    x = Dense(32, activation='relu')(x)

    output = Dense(1, activation='tanh', name='result')(x)
    model = Model(inputs=[pro, lig], output=output)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    return model


def cnn_train_generator(cnn_pro_train, cnn_lig_train):
    while True:
        for i in range(len(cnn_pro_train)):
            x1 = np.array([cnn_pro_train[i]])
            x2 = np.array([cnn_lig_train[i]])
            result = np.array([1])
            yield ({'pro': x1, 'lig': x2}, {'result': result})

            indexes_neg = sample_neg(NUM_NEG, i, NUM_TRAIN - 1)
            for sample in indexes_neg:
                x2 = np.array([cnn_lig_train[sample]])
                result = np.array([-1])
                yield ({'pro': x1, 'lig': x2}, {'result': result})
                # yield [np.array(cnn_pro_train[i]), np.array(cnn_lig_train[sample])]
    pass


def cnn_valid_generator(cnn_pro_valid, cnn_lig_valid):
    while True:
        for i in range(len(cnn_pro_valid)):
            for j in range(len(cnn_lig_valid)):
                x1 = np.array([cnn_pro_valid[i]])
                x2 = np.array([cnn_lig_valid[j]])
                if i ==j :
                    result = np.array([1])
                    yield ({'pro': x1, 'lig': x2}, {'result': result})
                else:
                    result = np.array([-1])
                    yield ({'pro': x1, 'lig': x2}, {'result': result})
    pass


def cnn_test_generator(cnn_pro_test, cnn_lig_test):
    while True:
        for i in range(len(cnn_pro_test)):
            for j in range(len(cnn_lig_test)):
                x1 = np.array([cnn_pro_test[i]])
                x2 = np.array([cnn_lig_test[j]])
                yield ({'pro': x1, 'lig': x2})
    pass


def lets_train_cnn(filename):
    with open(PATH_cnn_pro_train, 'rb') as f:
        cnn_pro_train = np.array(pickle.load(f))
    with open(PATH_cnn_lig_train, 'rb') as f:
        cnn_lig_train = np.array(pickle.load(f))
    gen_train = cnn_train_generator(cnn_pro_train[:NUM_TRAIN], cnn_lig_train[:NUM_TRAIN])

    with open(PATH_cnn_pro_train, 'rb') as f:
        cnn_pro_train = np.array(pickle.load(f))
    with open(PATH_cnn_lig_train, 'rb') as f:
        cnn_lig_train = np.array(pickle.load(f))

    gen_valid = cnn_valid_generator(cnn_pro_train[-NUM_VALID:], cnn_lig_train[-NUM_VALID:])



    cnn_out_train = np.array([[1], [-1], [-1]]*NUM_TRAIN)
    cnn_out_valid = np.eye(NUM_VALID).reshape(pow(NUM_VALID, 2),1)

    print(type(next(gen_train)[0]))

    model = model_3dcnn([27, 27, 27, 1], [6, 6, 6, 1])

    history = model.fit_generator(gen_train, steps_per_epoch=NUM_TRAIN*3, epochs=EPOCH, validation_data=gen_valid, validation_steps=NUM_TRAIN*3)

    model.save_weights(filename)

    # plot loss
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, EPOCH + 1), train_loss, label='train_loss')
    plt.plot(np.arange(1, EPOCH + 1), valid_loss, label='valid_loss')
    plt.title('Loss vs Epochs in Training and Validation Set for 3D-CNN')
    plt.xlabel('Epochs')
    plt.ylabel('Loss(MSE)')

    x_label = range(1, 11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    plt.savefig('../figs/3dcnn_validation.jpg', dpi=200)


def lest_valid_cnn(filename):
    with open(PATH_cnn_pro_train, 'rb') as f:
        cnn_pro_train = np.array(pickle.load(f))
    with open(PATH_cnn_lig_train, 'rb') as f:
        cnn_lig_train = np.array(pickle.load(f))

    gen_valid = cnn_test_generator(cnn_pro_train[-NUM_VALID:], cnn_lig_train[-NUM_VALID:])

    model = model_3dcnn([27, 27, 27, 1], [6, 6, 6, 1])
    model.load_weights(filename)

    predicted_cnn = model.predict_generator(gen_valid, steps=pow(NUM_VALID, 2)).reshape(NUM_VALID, NUM_VALID)
    print(predicted_cnn)

    # save as txt file
    result = []
    for i in range(len(predicted_cnn)):
        a = np.array(predicted_cnn[i, :])
        line = (heapq.nlargest(2, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)
    np.savetxt('../data/result/valid_predictions_3dcnn.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')

    # create correct answer for validation data
    result = []
    for i in range(300):
        tmp = []
        tmp.append(int(i + 1))
        tmp.append(int(i + 1))
        result.append(tmp)
    np.savetxt('../data/result/test_ground_truth_example.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig_id', fmt='%d')


def lest_test_cnn(filename):
    with open(PATH_cnn_pro_test, 'rb') as f:
        cnn_pro_train = np.array(pickle.load(f))
    with open(PATH_cnn_lig_test, 'rb') as f:
        cnn_lig_train = np.array(pickle.load(f))

    gen_valid = cnn_test_generator(cnn_pro_train[:NUM_TEST], cnn_lig_train[:NUM_TEST])

    model = model_3dcnn([27, 27, 27, 1], [6, 6, 6, 1])
    model.load_weights(filename)

    predicted_cnn = model.predict_generator(gen_valid, steps=pow(NUM_VALID, 2)).reshape(NUM_VALID, NUM_VALID)
    print(predicted_cnn)

    # save as txt file
    result = []
    for i in range(len(predicted_cnn)):
        a = np.array(predicted_cnn[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)
    np.savetxt('../data/result/test_predictions_3dcnn.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')


if __name__ == '__main__':


    print('lets train cnn！')
    lets_train_cnn('../model/3d-cnn.h5')

    # print('lets valid cnn！')
    # lest_valid_cnn('../model/3d-cnn.h5')

    print('lets test cnn！')
    lest_test_cnn('../model/3d-cnn.h5')






    '''
    model = model_3dcnn([27,27,27,1],[6,6,6,1])

    hist = model.fit(x=[cnn_pro_train, cnn_lig_train], y=cnn_out_train, epochs=1, verbose=1,)
                     #validation_data=([cnn_pro_valid, cnn_lig_valid], cnn_out_valid))




    preprocess_test.create_CNN_test(0, 824)
    with open('../data/cnn_data/cnn_pro_test.bin', 'rb') as f:
        cnn_pro_test = np.array(pickle.load(f))
    with open('../data/cnn_data/cnn_lig_test.bin', 'rb') as f:
        cnn_lig_test = np.array(pickle.load(f))

    # a = model.predict([cnn_pro_test, cnn_lig_test])
    # np.savetxt('../data/cnn_data/result.txt', a)'''



    '''
    ###########################################################################
    # get the prediction result of testing data set
    predicted_cnn = model.predict([cnn_pro_test,cnn_lig_test], batch_size=1)
    predicted_cnn = predicted_cnn.reshape(824, 824)
    print('The result is:')
    print(predicted_cnn)

    # save as txt file
    result = []
    for i in range(len(predicted_cnn)):
        a = np.array(predicted_cnn[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)
    np.savetxt('../data/result/test_predictions_cnn.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')
    '''