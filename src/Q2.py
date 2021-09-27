import scipy.io
import numpy as np
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt

#==============================LOAD spamData.mat FILE====================================
data = scipy.io.loadmat("spamData.mat")
print(data.keys())

#Training dataset
xtrain = np.array(data['Xtrain'])
ytrain = np.array(data['ytrain'])

#Testing dataset
xtest = np.array(data['Xtest'])
ytest = np.array(data['ytest'])

#Data processing - log transform each feature 
xtrain = np.log(xtrain+0.1)
xtest = np.log(xtest+0.1)

N = len(ytrain) #Total Number of email from training data
N1 = int(np.sum(ytrain, axis = 0)) #Number of spam email from trainng data(label = 1) 
N0 = N-N1 #Number of nonspam email from trainng data(label = 0) 

#=====================================FUNCTIONS=============================================

#Calculate class label prior using MLE
def calculatemle(label):
    mle = N1/N #MLE for label = 1
    if label == 0:
        mle = 1 - mle 
    return np.log(mle)

#Calculate feature likelihood using MLE estimates of mean and variance 
def calculatefl(label,feature_idx,feature): 
    xmean = mean_list[label][feature_idx]
    xvar = var_list[label][feature_idx]
    fl = -np.log(np.sqrt(2*np.pi*xvar))-((feature-xmean)**2/(2*xvar))
    return fl

#Classify data sample
def naivebayes(sample):
    predict_class_list = []
    for label in labels: #Calculate Posterior Prediction for Y = 1 and Y = 0 
        features_len = len(xtrain[0]) #number of features 
        predict_class = calculatemle(label) #Class label prior
        for feature_idx in range (features_len): #Feature likelihood - for all features compute p(x|y,theta)
            feature = sample[feature_idx] 
            predict_class += calculatefl(label,feature_idx,feature) #P(Y|X,D) = Class label prior + Feature Likelihood 
        predict_class_list.append(predict_class)
    max_value = max(predict_class_list) 
    idx = predict_class_list.index(max_value)
    return labels[idx]

#=========================================MAIN==================================================
labels = [0,1]
error_train_l = []
error_test_l = []

#Calculate mean and variance List
"""
Matrix with shape(2 x 57)
Each row for each class (Y=0 or Y=1)
Each column showing the mean and variance of the feature in that label
"""

mean_list = []
var_list = []
for label in labels:
    num_features = len(xtrain[0])
    xtrain_label = xtrain[ytrain.flatten()==label] #Data samples for Label Y = 1 or Y= 0 
    mean_list.append(xtrain_label.mean(axis=0))
    var_list.append(xtrain_label.var(axis=0))

#Predict target class of email data x using ML estimates of mean and variance 
"""Using MLE for class prior and feature likelihood """

#Predict y label for each data sample from training dataset 
training_results = []
for sample in xtrain: 
    predict_y = naivebayes(sample)
    training_results.append(predict_y) #Label prediction for each data sample in training dataset

#Predict y label for each data sample from testing dataset 
testing_results = []
for sample_test in xtest: 
    predict_y_test = naivebayes(sample_test)
    testing_results.append(predict_y_test) #Label prediction for each data sample in training dataset


#Calculate training error - Compare predicted label vs true label for each data sample
error_count_train= 0
for idx in range(len(training_results)):
    if (training_results[idx] != (ytrain.flatten())[idx]):
        error_count_train +=1
error_train_rate = error_count_train/len(xtrain) 
error_train_l.append(error_train_rate)
print("Error rate for training dataset:", error_train_rate)


#Calculate testing error - Compare predicted label vs true label for each data sample
error_count_test = 0
for idx in range(len(testing_results)):
    if (testing_results[idx] != (ytest.flatten())[idx]):
        error_count_test +=1
error_test_rate = error_count_test/len(xtest) 
error_test_l.append(error_test_rate)
print("Error rate for testing dataset:", error_test_rate)
print("\n")
    