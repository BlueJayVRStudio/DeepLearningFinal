import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix
from scipy import stats

from matplotlib.ticker import MaxNLocator

# Helper function from CSCA5622 Week 6: SVM Lab 
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
        
# Helper function from CSCA5622 Week 6: SVM Lab 
def plotSearchGrid(grid):
    
    scores = [x for x in grid.cv_results_["mean_test_score"]]
    scores = np.array(scores).reshape(len(grid.param_grid["C"]), len(grid.param_grid["gamma"]))

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(grid.param_grid["gamma"])), grid.param_grid["gamma"], rotation=45)
    plt.yticks(np.arange(len(grid.param_grid["C"])), grid.param_grid["C"])
    plt.title('Validation accuracy')
    plt.show()

# sklearn's confusion matrix: 
# i-th row = known class
# j-th column = predicted class

# precision = TP / (TP + FP)
# how much of the specific class prediction were true predictions 
def precision(confusion_mat):
    precisions = []
    for class_type in range(len(confusion_mat)):
        true_positive = confusion_mat[class_type][class_type]
        predictions = sum(confusion_mat[:, class_type])
        precisions.append(true_positive / predictions)
    return sum(precisions)/len(precisions)
    
# recall = TP / (TP + FN)
# how much of all true positives were correctly identified
def recall(confusion_mat):
    recalls = []
    for class_type in range(len(confusion_mat)):
        true_positive = confusion_mat[class_type][class_type]
        positives = sum(confusion_mat[class_type, :])
        recalls.append(true_positive/ positives)
    return sum(recalls)/len(recalls)


def plot_accuracy(hist, name):
    hist_train = hist.history['accuracy']
    hist_val = hist.history['val_accuracy']

    plt.figure(figsize=(11, 7))

    plt.plot(range(len(hist_train)), hist_train, marker='o', label = 'accuracy, training')
    plt.plot(range(len(hist_train)), hist_val, marker='o', label = 'accuracy, validation')
    
    plt.xticks(range(len(hist_train)), range(len(hist_train)), rotation=45)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))
    
    plt.legend() 
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(name)
    plt.grid(True)
    plt.show()

def plot_roc(y_true, y_pred, name):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Chance')
    plt.title(name)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()

def XOR(a, b):
    return (a and not b) or (not a and b)
    
# Summary Statistics Handling
def handle_stats(summary_stats, predictor, _x_test, _y_test, model_name):
    prediction = predictor.predict(_x_test).reshape((60, ))
    prediction = (prediction > 0.5).astype(int)
    ### Prepare labels
    labels = np.array(_y_test)

    print(f"Prediction: {prediction}, shape: {prediction.shape} \n")
    print(f"True Label: {labels}, shape: {labels.shape} \n")
    
    # Calculate Accuracy
    total = len(np.array(prediction == labels))
    correct = len(np.where(np.array(prediction == labels) == True)[0])
    accuracy = correct/total
    
    # Construct confusion matrix
    confusion_mat = confusion_matrix(labels, prediction)
    
    # Calculate Precision
    _precision = precision(confusion_mat)
    # Calculate Recall
    _recall = recall(confusion_mat)

    # Calculate p-value assuming binomial probability distribution for n = |label|
    p = 0.5
    n = len(labels)
    mean = p * n
    std = np.sqrt(n * p * (1-p))
    observed_accuracy = accuracy * n
    
    # Calculate the z-score
    z_score = (observed_accuracy - mean) / std
    
    # Calculate the p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    print(f"Z-Score: {z_score} \n")
    print(f"P-Value: {p_value} \n")
    
    summary_stats['P-Value'].append(p_value)
    summary_stats['Is Significant'].append(True) if p_value < 0.05 else summary_stats['Is Significant'].append(False)
    
    summary_stats['Model'].append(model_name)

    print(confusion_mat)

    ###    
    print(f"test accuracy: {accuracy}")
    print(f"test precision: {_precision}")
    print(f"test recall: {_recall}")

    summary_stats[f'Test Accuracy'].append(accuracy)
    summary_stats[f'Test Precision'].append(_precision)
    summary_stats[f'Test Recall'].append(_recall)


if __name__ == "__main__":
    print("helper functions for CSCA 5642 final project")

    # confusion_mat = np.array([np.array([4, 0, 1, 3]), 
    #                           np.array([0, 5, 3, 1]),
    #                           np.array([0, 0, 7, 0]),
    #                           np.array([1, 0, 0, 7])])
    # print(confusion_mat)
    # print(precision(confusion_mat))
    # print(recall(confusion_mat))