import scipy.io
import numpy as np
from sklearn.preprocessing import binarize
import matplotlib.pyplot as plt
  
#==============================LOAD spamData.mat FILE====================================
data = scipy.io.loadmat("spamData.mat")
print(data.keys())

#Trainin dataset
xtrain = np.array(data['Xtrain'])
ytrain = np.array(data['ytrain'])

#Testing dataset
xtest = np.array(data['Xtest'])
ytest = np.array(data['ytest'])

#Data processsing - Binarize feature data
xtrain = binarize(xtrain)
xtest = binarize(xtest)

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

#Calculate feature likelihood using posterior predictive of binary feature 
def calculatefl(label,feature_idx,feature,alpha):  
    if label ==0:
        N_y0f1 = Fnum_list[0][feature_idx] #Number of data samples from label 0 with feature = 1 for jth feature 
        fl = (N_y0f1+alpha)/ (N0 + (2*alpha))

    else:
        N_y1f1 = Fnum_list[1][feature_idx] #Number of data samples from label 1 with feature = 1 for feature j
        fl = (N_y1f1+alpha)/ ((N1 + (2*alpha)))

    if feature ==0:
        return np.log(1-fl)
    else:
        return np.log(fl)   

#Classify data sample
def naivebayes(sample,alpha):
    predict_class_list = []
    for label in labels: #Calculate Posterior Prediction for Y = 1 and Y = 0 
        features_len = len(xtrain[0]) #number of features 
        predict_class = calculatemle(label) #Class label prior
        for feature_idx in range (features_len): #Feature likelihood - for all features compute p(x|y,D)
            feature = sample[feature_idx] 
            predict_class += calculatefl(label,feature_idx,feature,alpha) #P(Y|X,D) = Class label prior + Feature Likelihood 
        predict_class_list.append(predict_class)
    max_value = max(predict_class_list)
    idx = predict_class_list.index(max_value)
    return labels[idx]

#=========================================MAIN==================================================
alphas = np.arange(0,100.5,0.5)
labels = [0,1]
error_train_l = []
error_test_l = []

#Calculate Feature Number List
"""
Matrix with shape(2 x 57)
Each row for each class (Y=0 or Y=1)
Each column showing the number of data samples with feature = 1 that label
"""
Fnum_list = []
for label in labels:
    num_features = len(xtrain[0])
    xtrain_label = xtrain[ytrain.flatten()==label] #Data samples for Label Y = 1 or Y= 0 
    Fnum_list.append(np.sum(xtrain_label,axis = 0)) #For each label, number of samples with feature =  1 for each feature  
print("Feature number list:",Fnum_list,"\n") 

#Predict target class of email data x using posterior predictive distribution 
"""Using MLE for class prior and posterior predictive for binary feature likelihood """

for alpha in alphas:
    
    #Predict y label for each data sample from training dataset 
    training_results = []
    for sample in xtrain: 
        predict_y = naivebayes(sample,alpha)
        training_results.append(predict_y) #Label prediction for each data sample in training dataset

    #Predict y label for each data sample from testing dataset 
    testing_results = []
    for sample_test in xtest: 
        predict_y_test = naivebayes(sample_test,alpha)
        testing_results.append(predict_y_test) #Label prediction for each data sample in training dataset

    #Calculate training error - Compare predicted label vs true label for each data sample
    error_count_train= 0
    for idx in range(len(training_results)):
        if (training_results[idx] != (ytrain.flatten())[idx]):
            error_count_train +=1
    error_train_rate = error_count_train/len(xtrain) 
    error_train_l.append(error_train_rate)
  
    #Calculate testing error - Compare predicted label vs true label for each data sample
    error_count_test = 0
    for idx in range(len(testing_results)):
        if (testing_results[idx] != (ytest.flatten())[idx]):
            error_count_test +=1
    error_test_rate = error_count_test/len(xtest) 
    error_test_l.append(error_test_rate)


    if (alpha == 1):
        print("Training error rate for alpha 1:", error_train_rate)
        print("Testing error rate for alpha 1:", error_test_rate)
    if (alpha == 10):
        print("Training error rate for alpha 10:", error_train_rate)
        print("Testing error rate for alpha 10:", error_test_rate)
    if (alpha == 100):
        print("Training error rate for alpha 100:", error_train_rate)
        print("Testing error rate for alpha 100:", error_test_rate)


#Plot training and test error rates vs alpha
plt.title("Q1. Beta-binomial Naive Bayes")
plt.plot(alphas,error_train_l,'g',label ="Training",linewidth = 2)
plt.plot(alphas,error_test_l,'r',label ="Testing",linewidth = 2)
plt.xlabel("Alpha")
plt.ylabel("Error Rates")
plt.legend()
plt.grid(True)
plt.show()













































































    









