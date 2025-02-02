import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    print("helper functions for CSCA 5642 final project")

    # confusion_mat = np.array([np.array([4, 0, 1, 3]), 
    #                           np.array([0, 5, 3, 1]),
    #                           np.array([0, 0, 7, 0]),
    #                           np.array([1, 0, 0, 7])])
    # print(confusion_mat)
    # print(precision(confusion_mat))
    # print(recall(confusion_mat))