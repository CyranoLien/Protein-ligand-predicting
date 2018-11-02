# Protein-ligand-predicting


CS5242 Neural Network and Deep Learning  
    
Team 23: Lian Yiming & Yan Maitong  


## Step 1: Run preprocess_train.py & preprocess_test.py

For training data,  
The first python file will call `store_tree()` function to generate the `tree_list.bin` in in `data/middle_data` folder;  
then call `create_mlp_lstm_train()` function to get two files named `train_input.bin` and `train_output.bin` in `data/middle_data` folder;  
finally call `create_CNN_train(3000)` function to create cnn_pro_train.bin and `cnn_lig_train.bin` in `data/cnn_data` folder.  

For testing data,  
The secpnd python file will call `store_tree()` function to generate the `tree_list_test.bin` for training data in the `data/middle_data` folder;  
then call `create_CNN_test(824)` function to create `cnn_pro_test.bin` and `cnn_lig_test.bin` in `data/cnn_data` folder.  


## Step 3: Run model_MLP.py  
The python file will create the MLP model, finish validation and store performance info, and finally predict the test data.  
Will get the out `test_predictions_mlp.txt` in `data/result` folder.  


## Step 4: Run model_LSTM.py  
The python file will create the LSTM model, finish validation and store performance info, and finally predict the test data.  
Will get the out `test_predictions_lstm.txt` in `data/result` folder.  


## Step 5: Run model_3dCNN.py  
The python file will call `lets_train_cnn()` to store weights of trained model in `model` folder;  
then call `lets_valid_cnn()` for validatin;  
finally call `lets test_cnn()` for preidiction;  
Will get the out `test_predictions_3dcnn.txt` in `data/result` folder.  


## Step 6: Run vote.py for final output
Finally combine the results of three models, and output one file named `test_predictions.txt` as the final prediction.  