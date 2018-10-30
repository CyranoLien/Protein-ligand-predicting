import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.preprocessing import sequence
from MyFlatten import MyFlatten
from pandas.core.frame import DataFrame


def create_clique_data():
    # store_tree()
    with open('../data/middle_data/tree_list.bin', 'rb') as f:
        tree_list = pickle.load(f)
    print('Tree info loaded successfully!')

    # store_train_data(tree_list, N)
    # store_valid_data(tree_list, N)
    with open('../data/middle_data/train_input.bin', 'rb') as f:
        train_input = pickle.load(f)
    with open('../data/middle_data/train_output.bin', 'rb') as f:
        train_output = pickle.load(f)
    with open('../data/middle_data/valid_input.bin', 'rb') as f:
        valid_input = pickle.load(f)
    with open('../data/middle_data/valid_output.bin', 'rb') as f:
        valid_output = pickle.load(f)
    print('Data info loaded successfully!')

    new_tin = []
    new_tout = []
    new_vin = []
    new_vout = []

    # for i in 2700
    for i in range(len(train_input)):
        lig = train_input[i]
        temp = lig.reshape(len(lig)*20, 1)
        new_tin.append(temp)
        new_tout.append([train_output[i]])
        '''
        if i<10:
            print(lig)
            print('*'*20)
            print(temp)'''

    with open('../data/mlp_data/new_tin.bin', 'wb') as f:
        pickle.dump(new_tin, f)
    with open('../data/mlp_data/new_tout.bin', 'wb') as f:
        pickle.dump(new_tout, f)

    for i in range(len(valid_input)):
        lig = valid_input[i]
        temp = lig.reshape(len(lig) * 20, 1)
        new_vin.append(temp)
        new_vout.append([valid_output[i]])
    with open('../data/mlp_data/new_vin.bin', 'wb') as f:
        pickle.dump(new_vin, f)
    with open('../data/mlp_data/new_vout.bin', 'wb') as f:
        pickle.dump(new_vout, f)


def create_clique_test():
    pass


def model_mlp():
    model = Sequential()
    model.add(Masking(mask_value=-999, batch_input_shape=(1, 200, 1)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(MyFlatten())
    model.add(Dense(1, activation='tanh'))
    return model


def start_train(x_train, y_train, x_valid, y_valid):
    tin = sequence.pad_sequences(x_train, maxlen=200, padding='post', dtype=float, value=-999)
    tout = y_train.reshape(8100, 1)
    vin = sequence.pad_sequences(x_valid, maxlen=200, padding='post', dtype=float, value=-999)
    vout = y_valid.reshape(90000, 1)

    print(tout.shape)

    model = model_mlp()
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    hist = model.fit(x=tin, y=tout, epochs=1, batch_size=1, verbose=1, validation_data=(vin, vout))

    a = model.predict(vin, batch_size=1)

    # loss, accuracy = model.evaluate(x=vin, y=vout)
    # print('Test loss is {:.4f}'.format(loss))
    # print('Test accuracy is {:.4f}'.format(accuracy))

    return a


def plot_hist(df, name, attr):
    fig, ax = plt.subplots()
    ax.plot(df['epoch'], df[attr], label='Train')
    ax.plot(df['epoch'], df['val_'+attr], label='Test')

    plt.title('Comparison on training and test data for ' + name + '_' +attr)
    plt.xlabel('epochs')
    plt.ylabel(name)
    plt.legend()
    plt.tight_layout()

    plt.savefig('../figs/hist_' + name + '_' + attr + '.jpg')
    plt.clf()




if __name__ == '__main__':
    # create_clique_data()

    with open('../data/mlp_data/new_tin.bin', 'rb') as f:
        x_train = np.array(pickle.load(f))
    with open('../data/mlp_data/new_tout.bin', 'rb') as f:
        y_train = np.array(pickle.load(f))
    with open('../data/mlp_data/new_vin.bin', 'rb') as f:
        x_valid = np.array(pickle.load(f))
    with open('../data/mlp_data/new_vout.bin', 'rb') as f:
        y_valid = np.array(pickle.load(f))


    d = start_train(x_train, y_train, x_valid, y_valid)
    # df = DataFrame(data=d)
    # epoch = df.index.map(lambda x: x + 1)
    # df['epoch'] = epoch
    # plot_hist(df, 'mlp', 'acc')
    # plot_hist(df, 'mlp', 'loss')


