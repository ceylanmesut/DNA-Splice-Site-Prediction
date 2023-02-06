# -*- coding: utf-8 -*-

#%%

import argparse

from src.utils import Data_Reader_Processor
from models.cnn_baseline import CNN_Baseline
from models.cnn import Dilated_CNN
from models.rnn_baseline import RNN_Baseline
from models.lstm import LSTM_Model

import warnings
warnings.filterwarnings('ignore')

#%%

def main(args):
    
    # Data preprocessing parameters
    MODEL_NAME = args.model_name

    EXPERIMENT = args.experiment # Experiment type: HUMAN or WORM

    SAMPLE_AMOUNT_TRAIN = args.sample_amount_tr
    SAMPLE_AMOUNT_VAL = args.sample_amount_val
    KMERS = args.kmers

    LR = args.learning_rate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    OUTPUT_DIM = args.output_dim
    DILATION_RATE = args.dilation_rate
    
    data_loader = Data_Reader_Processor(EXPERIMENT)
    X_train, X_val, X_test, y_train, y_val, y_test, max_length, vocab_size = data_loader.process_datasets(SAMPLE_AMOUNT_TRAIN, SAMPLE_AMOUNT_VAL, KMERS)

    if MODEL_NAME == "CNN_Baseline" : 
        
        # Defining model
        cnn_model = CNN_Baseline(X_train, y_train, X_val, y_val, X_test, y_test, LR, EPOCHS, BATCH_SIZE, 
                                vocab_size, max_length, OUTPUT_DIM, DILATION_RATE)

        model, _ = cnn_model.train_model()
        cnn_model.save_model(model, MODEL_NAME)

    elif MODEL_NAME == "Dilated_CNN":
        
        # Defining model
        cnn_model = Dilated_CNN(X_train, y_train, X_val, y_val, X_test, y_test, LR, EPOCHS, BATCH_SIZE, 
                                vocab_size, max_length, OUTPUT_DIM, DILATION_RATE)
        
        model, _ = cnn_model.train_model()
        cnn_model.save_model(model, MODEL_NAME)

    elif  MODEL_NAME == "RNN_Baseline":
        
        # Defining model
        rnn_baseline = RNN_Baseline(X_train, y_train, X_val, y_val, X_test, y_test, LR, EPOCHS, BATCH_SIZE, 
                                vocab_size, max_length, OUTPUT_DIM, DILATION_RATE)
        
        model, _ = rnn_baseline.train_model()
        rnn_baseline.save_model(model, MODEL_NAME)
        
    elif  MODEL_NAME == "LSTM_Model":
        
        # Defining model
        lstm_model = LSTM_Model(X_train, y_train, X_val, y_val, X_test, y_test, LR, EPOCHS, BATCH_SIZE, 
                                vocab_size, max_length, OUTPUT_DIM, DILATION_RATE)
        
        model, _ = lstm_model.train_model()
        lstm_model.save_model(model, MODEL_NAME)
        
if __name__== "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", default="Dilated_CNN", type=str)
    parser.add_argument("--experiment", default="HUMAN", type=str)
    parser.add_argument("--sample_amount_tr", default=20000, type=int)
    parser.add_argument("--sample_amount_val", default=10000, type=int)
    parser.add_argument("--kmers", default=4, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--output_dim", default=100, type=int)
    parser.add_argument("--dilation_rate", default=2, type=int)
    
    args = parser.parse_args()
    
    main(args)
