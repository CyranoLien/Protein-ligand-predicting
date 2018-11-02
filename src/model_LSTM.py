import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from sklearn.metrics import mean_squared_error
from keras.optimizers import RMSprop, Adam
import heapq
from preprocess_train import *
from preprocess_test import *

# Create model
def create_lstm_model(state):
    model = Sequential()
    model.add(Masking(mask_value=-999, batch_input_shape=(1, 50, 4)))
    model.add(
        LSTM(20, stateful=state, return_sequences=False, batch_input_shape=(1, 50, 4)))
    model.add(Dense(1), activation='tanh')
    return model


if __name__ == '__main__':

    # Train the model first

    with open('../data/middle_data/train_input.bin', 'rb') as f:
        train_input = np.array(pickle.load(f))



    with open('../data/middle_data/train_output.bin', 'rb') as f:
        y_train = np.array(pickle.load(f))


    with open('../data/middle_data/tree_list.bin', 'rb') as f:
        tree_list = pickle.load(f)
    print('Tree info loaded successfully!')


    valid_input, y_valid = create_mlp_lstm_valid(tree_list, begin=2700, end=3000)


    with open('../data/middle_data/tree_list_test.bin', 'rb') as f:
         tree_list_test = pickle.load(f)
    print('Tree info loaded successfully!')

    test_input = create_mlp_lstm_test(tree_list_test)




    x_train =[]
    for i in range(len(train_input)):
        temp = train_input[i][0]
        for j in range(len(train_input[i])):
            temp = np.concatenate((temp, train_input[i][j]), axis=0)
        x_train.append(temp)
    x_train = np.array(x_train)

    x_valid = []
    for i in range(len(valid_input)):
        temp = valid_input[i][0]
        for j in range(len(valid_input[i])):
            temp = np.concatenate((temp, valid_input[i][j]), axis=0)
        x_valid.append(temp)
    x_valid = np.array(x_valid)

    x_test = []
    for i in range(len(test_input)):
        temp = test_input[i][0]
        for j in range(len(test_input[i])):
            temp = np.concatenate((temp, test_input[i][j]), axis=0)
        x_test.append(temp)
    x_test = np.array(x_test)

    # padding sequence
    x_train = sequence.pad_sequences(x_train, maxlen=50, padding='post', dtype=float, value=-999)
    x_valid = sequence.pad_sequences(x_valid, maxlen=50, padding='post', dtype=float, value=-999)
    x_test = sequence.pad_sequences(x_test, maxlen=50, padding='post', dtype=float, value=-999)
    y_valid = np.array(y_valid)

    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)
    print(y_valid.shape)


    # create model
    model = create_lstm_model(state=False)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam)
    history = model.fit(x=x_train, y=y_train, epochs=10, batch_size=1, verbose=1, validation_data=(x_valid, y_valid))

    # plot loss
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, 10 + 1), train_loss, label='train_loss')
    plt.plot(np.arange(1, 10 + 1), valid_loss, label='valid_loss')
    plt.title('Loss vs Epochs in Training and Validation Set for LSTM')
    plt.xlabel('Epochs')
    plt.ylabel('Loss(MSE)')
    x_label = range(1, 11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    plt.savefig('../figs/lstm_validation.jpg', dpi=200)


    # predict validation data
    predicted_lstm = model.predict(x_valid, batch_size=1)

    # create answer for validation data
    result = []
    for i in range(300):
        tmp = []
        tmp.append(int(i+1))
        tmp.append(int(i+1))
        result.append(tmp)
    np.savetxt('../data/result/test_ground_truth_example.txt', result, delimiter='\t', newline='\n', comments='',
                   header='pro_id\tlig_id', fmt='%d')


    # get the prediction result of testing data set
    predicted_lstm = predicted_lstm.reshape(300, 300)


    # save as txt file
    result = []
    for i in range(len(predicted_lstm)):
        a = np.array(predicted_lstm[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)

    np.savetxt('../data/result/test_predictions_lstm_example.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')
   
    ############################
    # predict testing data
    predicted_lstm = model.predict(x_test, batch_size=1)

    # get the prediction result of testing data set
    predicted_lstm = predicted_lstm.reshape(824, 824)

    # save as txt file
    result = []
    for i in range(len(predicted_lstm)):
        a = np.array(predicted_lstm[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)

    np.savetxt('../data/result/test_predictions_lstm.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')
