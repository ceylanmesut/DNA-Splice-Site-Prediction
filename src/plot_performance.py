# -*- coding: utf-8 -*-



from src.utils import Data_Reader_Processor
from sklearn import metrics
from src.utils import AUC_K, Precision_K
from tensorflow import keras

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import warnings
warnings.filterwarnings('ignore')

EXPERIMENT = "WORM" # Experiment type: HUMAN or WORM
SAMPLE_AMOUNT_TRAIN = 2000
SAMPLE_AMOUNT_VAL = 1000
KMERS = 4

data_loader = Data_Reader_Processor(EXPERIMENT)
X_train, X_val, X_test, y_train, y_val, y_test, max_length, vocab_size = data_loader.process_datasets(SAMPLE_AMOUNT_TRAIN, SAMPLE_AMOUNT_VAL, KMERS)

figure = plt.figure(figsize=(14,7))
ax1 = figure.add_subplot(121)
ax2 = figure.add_subplot(122)


worm_models = ["worm_models\\RNN_Baseline_worm.h5", "worm_models\\LSTM_model_worm.h5",
               "worm_models\\CNN_Baseline_worm.h5", "worm_models\\Dilated_CNN_worm.h5"]

human_models = ["human_models\\RNN_Baseline.h5", "human_models\\LSTM_model.h5",
               "human_models\\CNN_Baseline.h5", "human_models\\Dilated_CNN_Network.h5"]

objects = {"AUC_K": AUC_K, "AUC_K":AUC_K, "Precision_K": Precision_K}

for m in worm_models:
    
    model_name = m.split("\\")[-1]
    model_name = model_name.split(".")[0]
    print("Model Name:",model_name)
    
    model = keras.models.load_model(m, custom_objects=objects)
    preds = model.predict(X_test)

    #### Reporting Performance ####
    roc_auc = metrics.roc_auc_score(y_test, preds)
    print("Test AUCROC Score: %.3f"%(roc_auc))

    average_precision = metrics.average_precision_score(y_test, preds)
    print('Test AUC-AP score: %.3f' %(average_precision))
    
    predictions = preds.ravel()
    fpr, tpr, thresholds_keras = metrics.roc_curve(y_test, predictions)
    
    auc_to_report = metrics.auc(fpr, tpr)
    print("AUCROC %.3f" % (auc_to_report))
    
    precision, recall, _ = metrics.precision_recall_curve(y_test, predictions)
    average_precision = metrics.average_precision_score(y_test, predictions)
    print('AUCPR: %.3f' % (average_precision))
    label = 'AP=%.3f' % (average_precision)
    
    ax1.plot(fpr, tpr, label='{:s}:(AUC = {:.2f})'.format(model_name, auc_to_report))
    ax2.plot(recall, precision, label='{:s}:(AP = {:.2f})'.format(model_name, average_precision), lw=2, alpha=.8)

ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.set_title('ROC Curves')
ax1.legend(loc='best')

ax2.set_title("Precision-Recall Curves")
ax2.legend(loc="best")
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')

plt.savefig("results_worm.png", dpi=200)
