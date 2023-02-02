


import tensorflow as tf
from keras.layers import Conv1D, Dense, Flatten, Embedding
from keras.models import Sequential
import datetime
from sklearn import metrics
import sys
sys.path.append("..") 
from src.visualize import Visualizer
from src.utils import AUC_K, Precision_K

class CNN_Baseline:
    
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                 LR, epochs, bs, vocab_size, max_length, output_dim=2000, d_rate=2):
        
        # Input
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test 
        self.y_test = y_test
        
        # Model parameters
        self.lr = LR
        self.epochs = epochs
        self.batch_size = bs
        self.dilation_rate = d_rate
        
        # Input parameters
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.output_dim = output_dim
    
    def train_model(self):
        
        self.model = self._build_model(self.lr)
        history =  self._fit(self.model, self.epochs, self.batch_size)
        predictions = self._predicit(self.model)
        
        self._report_performance(history, predictions)
        
        return(self.model, history)

    def save_model(self, model, model_name=None):
      
        ext = ".h5"
        save_name = model_name + ext
        model.save(save_name)

    def _build_model(self, lr):
  
        tf.keras.backend.clear_session()

        #CNN APPROACH
        model = Sequential()
        # Embedding Layer
        model.add(Embedding(input_dim=self.vocab_size, output_dim=self.output_dim, input_length=self.max_length))
        
        # 1st Layer
        model.add(Conv1D(filters=256, kernel_size=2))

        # 2nd Layer
        model.add(Conv1D(filters=128, kernel_size=2))

        # 4th Layer
        model.add(Flatten())
        model.add(Dense(50))
        
        # Output Layer
        model.add(Dense(1, activation='sigmoid'))
        
        adam_opt = tf.keras.optimizers.Adam(learning_rate = lr)

        metrics_train=[
            AUC_K(name="AUCROC", curve="ROC", from_logits=True),
            AUC_K(name="AUCPR", curve="PR", from_logits=True),
            Precision_K(from_logits=True)]


        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=adam_opt, metrics=metrics_train)
        model.summary()    

        return model

    def _fit(self, model, epochs, batch_size):
        

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)]

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        history = model.fit(self.X_train, self.y_train, 
                        epochs=epochs, verbose=1, validation_data=(self.X_val, self.y_val), 
                        batch_size=batch_size, shuffle=False, callbacks=[tensorboard_callback, early_stopping_callback])
        
        return history
    
    def _predicit(self, model):
        
        preds = model.predict(self.X_test)
        
        return preds
    
    def _report_performance(self, history, preds):

        roc_auc = metrics.roc_auc_score(self.y_test, preds)
        print("Test AUCROC Score: %.3f"%(roc_auc))

        average_precision = metrics.average_precision_score(self.y_test, preds)
        print('Test AUC-AP score: %.3f' %(average_precision))
        
        visualizer = Visualizer()
        visualizer.plot_performance(history)
        visualizer.plot_roc_curves(self.model, self.X_test, self.y_test)