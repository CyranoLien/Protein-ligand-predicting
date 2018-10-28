x_train=np.array(x_train)
y_train=np.array(y_train)
x_valisation=np.array(x_valisation)
y_validation=np.array(y_validation)

x_train = sequence.pad_sequences(x_train, maxlen=15000, padding='post', dtype=float, value=-999)
x_valisation = sequence.pad_sequences(x_valisation, maxlen=15000, padding='post', dtype=float, value=-999)


# Create model
def create_lstm_model(stateful):
    ##### YOUR MODEL GOES HERE #####
    model = Sequential()
    model.add(Masking(mask_value=-999, batch_input_shape=(1, 15000, 4)))
    model.add(
        LSTM(20, stateful=stateful, return_sequences=False, batch_input_shape=(1, 15000, 4)))
    model.add(Dense(1, activation='sigmoid'))
    return model

batch_size=1
# Create the stateful model
print('Creating Stateful LSTM Model...')
model_lstm_stateless = create_lstm_model(stateful=False)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00005)
model_lstm_stateless.compile(optimizer=adam, loss='mean_squared_error')

model_lstm_stateless.fit(x_train, y_train, epochs=1, batch_size=batch_size, validation_data=(x_valisation, y_validation), shuffle=False)

predicted_lstm_stateless = model_lstm_stateless.predict(x_valisation, batch_size=batch_size)
lstm_stateless_rmse = np.sqrt(mean_squared_error(y_validation, predicted_lstm_stateless))
print('The rmse of testing data is:' + str(lstm_stateless_rmse))
print('The result is:')
print(predicted_lstm_stateless)
print(type(predicted_lstm_stateless))
predicted_lstm_stateless = predicted_lstm_stateless.reshape(501, 6)
print(predicted_lstm_stateless)

for i in range(len(predicted_lstm_stateless)):
    a = np.array(predicted_lstm_stateless[i,:])
    print(heapq.nlargest(3, range(len(a)), a.take))