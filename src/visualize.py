

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn import metrics


class Visualizer:
    
    def __init__(self):
        pass
    
    def plot_performance(self, history):

        plt.subplots(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss', fontsize = 10)
        plt.ylabel('Loss', fontsize = 10)
        plt.xlabel('Epoch', fontsize = 10)
        plt.legend(['Train', 'Validation'], fontsize = 10)

        plt.subplot(1, 3, 2)
        plt.plot(history.history['AUCPR'])
        plt.plot(history.history['val_AUCPR'])
        plt.title('ROC PR', fontsize = 10)
        plt.ylabel('ROC PR', fontsize = 10)
        plt.xlabel('Epoch', fontsize = 10)
        plt.legend(['Train', 'Validation'], fontsize =10)

        plt.subplot(1, 3, 3)
        plt.plot(history.history['AUCROC'])
        plt.plot(history.history['val_AUCROC'])
        plt.title('AUCROC', fontsize = 10)
        plt.ylabel('AUCROC', fontsize = 10)
        plt.xlabel('Epoch', fontsize = 10)
        plt.legend(['Train', 'Validation'], fontsize = 10)

        plt.tight_layout()
        plt.show()


    def plot_roc_curves(self, model, X_test, y_test):
        
        # AUCROC 
        predictions = model.predict(X_test).ravel()
        fpr, tpr, thresholds_keras = roc_curve(y_test, predictions)
        auc_to_report = auc(fpr, tpr)
        print("AUCROC %.3f" % (auc_to_report))

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Dilated CNN (AUC = {:.3f})'.format(auc_to_report))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()

        # AUCPR
        predictions_pr = model.predict(X_test).ravel()
        probabilities = predictions_pr
        precision, recall, _ = metrics.precision_recall_curve(y_test, probabilities)
        average_precision = metrics.average_precision_score(y_test, predictions)
        
        print('AUCPR: %.3f' % (average_precision))
        label = 'AP=%.3f' % (average_precision)

        plt.plot(recall, precision, label=label, lw=2, alpha=.8)
        plt.title("Precision-Recall Curve")
        plt.legend(loc="upper right")
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()