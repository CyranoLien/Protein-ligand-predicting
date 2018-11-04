import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Masking
from keras.preprocessing import sequence
from MyFlatten import MyFlatten
from preprocess_train import *
from preprocess_test import *

def create_clique_data():


    with open('../data/middle_data/train_input.bin', 'rb') as f:
        train_input = pickle.load(f)
    with open('../data/middle_data/train_output.bin', 'rb') as f:
        train_output = pickle.load(f)

    with open('../data/middle_data/tree_list.bin', 'rb') as f:
        tree_list = pickle.load(f)
    print('Tree info loaded successfully!')

    valid_input, valid_output = create_mlp_lstm_valid(tree_list, 2700)

    with open('../data/middle_data/tree_list_test.bin', 'rb') as f:
        tree_list_test = pickle.load(f)
    print('Tree info loaded successfully!')

    test_input = create_mlp_lstm_test(tree_list_test)

    new_train_input = []
    new_train_output = []
    new_valid_input = []
    new_valid_output = []
    new_test_input = []

    # for i in 2700
    for i in range(len(train_input)):
        lig = train_input[i]
        temp = lig.reshape(len(lig)*20, 1)
        new_train_input.append(temp)
        new_train_output.append([train_output[i]])

    for i in range(len(valid_input)):
        lig = valid_input[i]
        temp = lig.reshape(len(lig) * 20, 1)
        new_valid_input.append(temp)
        new_valid_output.append([valid_output[i]])


    for i in range(len(test_input)):
        lig = test_input[i]
        temp = lig.reshape(len(lig) * 20, 1)
        new_test_input.append(temp)

    return np.array(new_train_input), np.array(new_train_output), np.array(new_valid_input), np.array(new_valid_output),\
           np.array(new_test_input)

def model_mlp():
    model = Sequential()
    model.add(Masking(mask_value=-999, batch_input_shape=(1, 200, 1)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(MyFlatten())
    model.add(Dense(1, activation='tanh'))
    return model

if __name__ == '__main__':

    x_train, y_train, x_valid, y_valid, x_test = create_clique_data()

    x_train = sequence.pad_sequences(x_train, maxlen=200, padding='post', dtype=float, value=-999)
    y_train = y_train.reshape(8100, 1)
    x_valid = sequence.pad_sequences(x_valid, maxlen=200, padding='post', dtype=float, value=-999)
    y_valid = y_valid.reshape(90000, 1)
    x_test = sequence.pad_sequences(x_test, maxlen=200, padding='post', dtype=float, value=-999)

    model = model_mlp()

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    history = model.fit(x=x_train, y=y_train, epochs=10, batch_size=1, verbose=1, validation_data=(x_valid, y_valid))

    filename = '../model/mlp.h5'
    model.save_weights(filename)
    #model.load_weights(filename)

    # plot loss
    train_loss = history.history['loss']
    valid_loss = history.history['val_loss']
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, 10 + 1), train_loss, label='train_loss')
    plt.plot(np.arange(1, 10 + 1), valid_loss, label='valid_loss')
    plt.title('Loss vs Epochs in Training and Validation Set for MLP')
    plt.xlabel('Epochs')
    plt.ylabel('Loss(MSE)')
    x_label = range(1, 11)
    plt.xticks(x_label)
    plt.legend()
    plt.grid()
    plt.savefig('../figs/mlp_validation.jpg', dpi=200)
    
    #predict validation data
    predicted_mlp = model.predict(x_valid, batch_size=1)

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
    predicted_mlp = predicted_mlp.reshape(300, 300)

    # save as txt file
    result = []
    for i in range(len(predicted_mlp)):
        a = np.array(predicted_mlp[i, :])
        line = (heapq.nlargest(10, range(len(a)), a.take))
        # nlargest function returns value from 0, add 1 to change to start from 1
        nl = [i + 1 for i in line]
        nl.append(int(i + 1))
        nl.reverse()
        result.append(nl)

    np.savetxt('../data/result/test_predictions_mlp_example.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')

    ############################
    # predict testing data
    predicted_mlp = model.predict(x_test, batch_size=1)

    # get the prediction result of testing data set
    predicted_lstm = predicted_mlp.reshape(824, 824)

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

    np.savetxt('../data/result/test_predictions_mlp.txt', result, delimiter='\t', newline='\n', comments='',
               header='pro_id\tlig1_id\tlig2_id\tlig3_id\tlig4_id\tlig5_id\tlig6_id\tlig7_id\tlig8_id\tlig9_id\tlig10_id',
               fmt='%d')


