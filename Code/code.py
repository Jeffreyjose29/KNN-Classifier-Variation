# Required to 'see' dynamic plots in Juypter notebooks
%matplotlib notebook
from collections import Counter

#Importing various packages inorder for the implementation to run smoothly
from skmultiflow.bayes import NaiveBayes
from skmultiflow.data import FileStream
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.trees import HAT
from skmultiflow.drift_detection import ADWIN
from skmultiflow.data import WaveformGenerator
from skmultiflow.lazy import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDClassifier
from sklearn.base import TransformerMixin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier 
from sklearn.ensemble import RandomForestClassifier

from skmultiflow.core import ClassifierMixin
from skmultiflow.utils.utils import *

# This statement is given incorrectly in the provided code
from skmultiflow.lazy.base_neighbors import BaseNeighbors
from skmultiflow.data import FileStream
from skmultiflow.lazy import KNNClassifier

#Whats the difference between HoeffdingTreeClassifier and HoeffdingTree
from skmultiflow.trees import HoeffdingTreeClassifier, HoeffdingAdaptiveTreeClassifier, HoeffdingTree
from skmultiflow.trees import isoup_tree
from skmultiflow.transform.base_transform import StreamTransform
from skmultiflow.utils import FastBuffer, get_dimensions
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest

import math
import sys
import warnings
warnings.filterwarnings('ignore')

def KNN(n_neighbors=5, max_window_size=1000, leaf_size=30):     # pragma: no cover
    warnings.warn("'KNN' has been renamed to 'KNNClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)

    return KNNClassifier(n_neighbors=n_neighbors, max_window_size=max_window_size, leaf_size=leaf_size)


class KNNClassifier(BaseNeighbors, ClassifierMixin):
    #Its like a constructor
    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean', weighted_vote = False):
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size, leaf_size=leaf_size, metric=metric)
        self.classes = []
                
    def partial_fit(self, X, y, classes=None, sample_weight=None):

        r, c = get_dimensions(X)

        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        for i in range(r):
            self.data_window.add_sample(X[i], y[i])

        return self


    def predict(self, X):
        #Add standardization in here
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred
    

    #Modify this method
    def predict_proba(self, X):
        #Add standardization in this method too
        r, c = get_dimensions(X)
        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # The model is empty, defaulting to zero
            return np.zeros(shape=(r, 1))
        proba = []

        self.classes = list(set().union(self.classes, np.unique(self.data_window.targets_buffer.astype(np.int))))
        new_dist, new_ind = self._get_neighbors(X)

        for i in range(r):
            votes = [0.0 for _ in range(int(max(self.classes) + 1))]
            for index in new_ind[i]:
                votes[int(self.data_window.targets_buffer[index])] += 1. / len(new_ind[i])
            proba.append(votes)

        return np.asarray(proba)

class MyKNNClassifier(KNNClassifier): # ... 
    def __init__(self, n_neighbors=5, max_window_size=1000, leaf_size=30, metric='euclidean', weighted_vote=False,
                 standardize = False):
        self.weighted_vote = weighted_vote
        self.standardize = standardize
        super().__init__(n_neighbors=n_neighbors, max_window_size=max_window_size, leaf_size=leaf_size, metric=metric)
        self.window_size = max_window_size
        self.window = None

        self.__configure()

    def __configure(self):
        self.window = FastBuffer(max_size=self.window_size)
        
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if(self.standardize == True):
            instance = np.array(X)
            X = self.transform_vector(instance)
            self.window.add_element(X)
        r, c = get_dimensions(X)

        if classes is not None:
            self.classes = list(set().union(self.classes, classes))

        for i in range(r):
            self.data_window.add_sample(X[i], y[i])

        return self
    
    
    def standardization(self, X):   
        #scaler = MinMaxScaler(feature_range=(0, 1))
        #scaler = scaler.fit(X)
        #print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
        #normalize the dataset and print the first 5 rows
        #normalized = scaler.transform(X)
        
        #return X
        
        scaler = StandardScaler()
        scaler.fit(X)
        normalized = scaler.fit_transform(X)
        X = normalized
        
        return X
    

    #Modify this method
    def predict_proba(self, X):
        #print("Not Weighted")
        #Add standardization in this method too
        if(self.standardize == True):
            instance = np.array(X)
            X = self.transform_vector(instance)
            
        r, c = get_dimensions(X)

        #print("Value of R: ", r) # r = 1
        #print("Value of C: ", c) # c = 2
        if self.data_window is None or self.data_window.size < self.n_neighbors:
            # The model is empty, defaulting to zero
            return np.zeros(shape=(r, 1))
        proba = []

        self.classes = list(set().union(self.classes, np.unique(self.data_window.targets_buffer.astype(np.int))))
        new_dist, new_ind = self._get_neighbors(X)

        #print("new_dist: ", new_dist)
        #print("new_ind: ", new_ind)
        ###################################### Weighting that I've added #######################################################
        #if(self.weighted_vote == True):
            #votes = self.vote(new_ind)
        #  self.classes = int(self.data_window.get_targets_matrix()[new_ind]) #Class of our index
        
        if(self.weighted_vote == False):
            #print("Not Weighted")
            for i in range(r):
                votes = [0.0 for _ in range(int(max(self.classes) + 1))]
                for index in new_ind[i]:
                    votes[int(self.data_window.targets_buffer[index])] += 1. / len(new_ind[i])

                proba.append(votes)
                
        else:
            #print("Weighted")
            position = 0
            for i in range(r):
                votes = [0.0 for _ in range(int(max(self.classes) + 1))]
                for index in new_ind[i]:
                    votes[int(self.data_window.targets_buffer[index])] += np.sum((1. / new_dist[i][position])) / len(new_ind[i])
                    position = position + 1
                proba.append(votes)

        return np.asarray(proba)

    
    def calculate_mean(self, column_index):
        mean = 0.
        if not self.window.is_empty():
            mean = np.nanmean(np.array(self.window.get_queue())[:, column_index])
        return mean

    def calculate_stddev(self, column_index):
        std = 1.
        if not self.window.is_empty():
            std = np.nanstd(np.array(self.window.get_queue())[:, column_index])
        if(std == 0.):
            std = 1.
        return std
    
    def transform_vector(self, X):
        r, c = get_dimensions(X)
        for i in range(r):
            row = np.copy([X[i][:]])
            for j in range(c):
                value = X[i][j]
                mean = self.calculate_mean(j)
                standard_deviation = self.calculate_stddev(j)
                standardized = (value - mean) / standard_deviation
                X[i][j] = standardized
            self.window.add_element(row)
        return X

#Read the stream 
stream = FileStream("C:/Users/jeffr/OneDrive/Desktop/Data Stream/Assignment_One/dataset/data_n30000.csv")
stream.prepare_for_use()

#stream.next_sample(10)
#stream.n_remaining_samples()
#X, y = stream.next_sample(5000)

metrics = ['accuracy', 'kappa', 'kappa_m', 'kappa_t', 'running_time', 'model_size'] 
evaluator = EvaluatePrequential(max_samples = 30000, n_wait = 100, show_plot = True, metrics = metrics) 

my_knn = MyKNNClassifier(standardize = True, weighted_vote = False)
evaluator.evaluate(stream = stream, model = [my_knn], model_names = ['My_KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix
print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

#All the methods that we need to test
knn = KNNClassifier()
ht = HoeffdingTreeClassifier(leaf_prediction = 'mc')
htnb = HoeffdingTreeClassifier(leaf_prediction = 'nb')
nb = NaiveBayes()
hoef = HoeffdingTreeClassifier()

#Evaluating all methods together
evaluator.evaluate(stream = stream, model = [knn, ht, htnb, nb, hoef], model_names = ['KNN', 'HTMC', 'HTNB', 'NB', 'HT'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

k_values = [1, 3, 20]
window_sizes = [15, 600, 6000]

evaluator = EvaluatePrequential(max_samples=30000, n_wait=100, show_plot=False, metrics=metrics)

for k in k_values:
    for w in window_sizes:
        myknnAltModel = MyKNNClassifier(standardize = False, weighted_vote = False, max_window_size=w, n_neighbors=k)
        evaluator.evaluate(stream=stream, model=[myknnAltModel], model_names=[("KNN k=" + str(k) + " w=" + str(w))])
        cm = evaluator.get_mean_measurements(0).confusion_matrix
        print("Recall per class")
        for i in range(cm.n_classes):
            recall = cm.data[(i,i)]/cm.sum_col[i] \
            if cm.sum_col[i] != 0 else 'Ill-defined'
            print("Class {}: {}".format(i, recall))
        print("\n")

k_values = [1, 3, 20]
window_sizes = [15, 600, 6000]

evaluator = EvaluatePrequential(max_samples=30000, n_wait=100, show_plot=False, metrics=metrics)

for k in k_values:
    for w in window_sizes:
        myknnAltModel = MyKNNClassifier(standardize = False, weighted_vote = True, max_window_size=w, n_neighbors=k)
        evaluator.evaluate(stream=stream, model=[myknnAltModel], model_names=[("KNN k=" + str(k) + " w=" + str(w))])
        cm = evaluator.get_mean_measurements(0).confusion_matrix
        print("Recall per class")
        for i in range(cm.n_classes):
            recall = cm.data[(i,i)]/cm.sum_col[i] \
            if cm.sum_col[i] != 0 else 'Ill-defined'
            print("Class {}: {}".format(i, recall))
        print("\n")

k_values = [1, 3, 20]
window_sizes = [15, 600, 6000]

evaluator = EvaluatePrequential(max_samples=30000, n_wait=100, show_plot=False, metrics=metrics)

for k in k_values:
    for w in window_sizes:
        myknnAltModel = MyKNNClassifier(standardize = True, weighted_vote = False, max_window_size=w, n_neighbors=k)
        evaluator.evaluate(stream=stream, model=[myknnAltModel], model_names=[("KNN k=" + str(k) + " w=" + str(w))])
        cm = evaluator.get_mean_measurements(0).confusion_matrix
        print("Recall per class")
        for i in range(cm.n_classes):
            recall = cm.data[(i,i)]/cm.sum_col[i] \
            if cm.sum_col[i] != 0 else 'Ill-defined'
            print("Class {}: {}".format(i, recall))
        print("\n")


stream_ = FileStream("C:/Users/jeffr/OneDrive/Desktop/Data Stream/Assignment_One/dataset/covtype_numeric.csv")
stream_.prepare_for_use()

my_knn1 = MyKNNClassifier(standardize = True, weighted_vote = False)
my_knn1

metrics = ['accuracy', 'kappa', 'kappa_m', 'kappa_t', 'running_time', 'model_size'] 
evaluator = EvaluatePrequential(max_samples = 30000, n_wait = 100, show_plot = True, metrics = metrics) 

evaluator.evaluate(stream = stream_, model = [my_knn1], model_names = ['My_KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

knn_data_ = KNNClassifier()
knn_data_

#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [knn_data_], model_names = ['KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))


knn_2 = KNNClassifier(n_neighbors = 10)
knn_2

#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [knn_2], model_names = ['KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

k_weighted = MyKNNClassifier(n_neighbors = 10, weighted_vote = True)
k_weighted

evaluator.evaluate(stream = stream_, model = [k_weighted], model_names = ['KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

ht_data_ = HoeffdingTreeClassifier(leaf_prediction = 'mc')
#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [ht_data_], model_names = ['HT-MC'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

htnb_data_ = HoeffdingTreeClassifier(leaf_prediction = 'nb')

#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [htnb_data_], model_names = ['HT-NB'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

nb_data_ = NaiveBayes()

#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [nb_data_], model_names = ['NB'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

ht_default = HoeffdingTreeClassifier()

#Evaluating all methods together
evaluator.evaluate(stream = stream_, model = [ht_default], model_names = ['HT Default'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

std_knn = MyKNNClassifier(n_neighbors = 10, standardize = True)
std_knn
evaluator.evaluate(stream = stream_, model = [std_knn], model_names = ['KNN'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

hatc = HoeffdingAdaptiveTreeClassifier()
evaluator.evaluate(stream = stream_, model = [hatc], model_names = ['Adaptive Tree'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))
    
    ooza = OzaBaggingAdwin()
evaluator.evaluate(stream = stream_, model = [ooza], model_names = ['Oza Bagging Adwin'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))

arf = AdaptiveRandomForest()
evaluator.evaluate(stream = stream_, model = [arf], model_names = ['Adaptive Random Forest'])
cm = evaluator.get_mean_measurements(0).confusion_matrix

print("Recall per class")
for i in range(cm.n_classes):
    recall = cm.data[(i,i)]/cm.sum_col[i] \
    if cm.sum_col[i] != 0 else 'Ill-defined'
    print("Class {}: {}".format(i, recall))