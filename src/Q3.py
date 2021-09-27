import scipy.io
from scipy.special import expit
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

#================================FUNCTIONS==============================================

def derivatives(xtrain_concat,ytrain, W_,lamda):
    #Calculate grep
    """greg(w) = g(w) + lamda*w"""

    µ_i = expit(np.dot(xtrain_concat,W_))
    g_W = np.dot(xtrain_concat.transpose(), (µ_i - ytrain))
    ω = np.concatenate((np.zeros((1, 1)), W_[1:]), axis=0) #Exclude bias from l2 regulatisation - W without the first element 
    greg_W = g_W + np.dot(lamda,ω)

    #Calculate Hreg
    """Hreg(w) = H(w) + lamda*I"""
    Sdiag = []
    for µ in µ_i:
        Sdiag.append(µ[0]*(1-µ[0]))
    S = np.diag(Sdiag)
    H_W = np.dot(xtrain_concat.transpose(), np.dot(S,xtrain_concat))

    I = np.identity(D + 1) 
    I[0,0] = 0 #first row and column are zeroes 
    Hreg_W = H_W + np.dot(lamda,I)

    return greg_W,Hreg_W

def newton_method(xtrain_concat,ytrain,lamba):
    W_ = np.zeros(((D+1),1)) #(D+1) x 1 Vector 
    step = True
    while step == True: 
        greg, Hreg = derivatives(xtrain_concat,ytrain,W_,lamda)
        temp = np.dot(np.linalg.inv(Hreg), greg)
        W_next = W_ - temp
        
        check = np.sum(np.abs(temp))
        if check > 0.0001:
            W_ = W_next
        else:
            step = False
    return W_


#==================================MAIN=================================================

lamda_1 = np.arange(1,11)
lamda_2 = np.arange(15,105,5)
lamda_list = np.concatenate((lamda_1,lamda_2),axis = 0)
print(lamda_list)

#concatenate bias (1) to strart of x_i
a = np.ones((xtrain.shape[0], 1))
b = np.ones((xtest.shape[0], 1))
xtrain_concat = np.concatenate((a, xtrain), axis = 1)
xtest_concat = np.concatenate((b, xtest), axis = 1)
 
D = xtrain.shape[1] #Feature Vector Length 
error_train_l = []
error_test_l = []

for lamda in lamda_list:

    W = newton_method(xtrain_concat,ytrain,lamda)

    #Predict results for each data sample from training dataset 
    training_results = []
    p_training = expit(np.dot(xtrain_concat,W))
    training_results  = p_training > 0.5

    #Calculate training error - Compare predicted label vs true label for each data sample
    error_count_train= 0
    for idx in range(len(training_results)):
        if (training_results[idx][0] != (ytrain.flatten())[idx]):
            error_count_train +=1
    error_train_rate = error_count_train/len(xtrain) 
    error_train_l.append(error_train_rate)
   

    #Predict results for each data sample from testing dataset 
    testing_results = []
    p_testing= expit(np.dot(xtest_concat,W))
    testing_results  = p_testing > 0.5

    #Calculate testing error - Compare predicted label vs true label for each data sample
    error_count_test = 0
    for idx in range(len(testing_results)):
        if (testing_results[idx][0] != (ytest.flatten())[idx]):
            error_count_test +=1
    error_test_rate = error_count_test/len(xtest) 
    error_test_l.append(error_test_rate)
    

    if (lamda == 1):
        print("Training error rate for lamda 1:", error_train_rate)
        print("Testing error rate for lamda 1:", error_test_rate)
    if (lamda == 10):
        print("Training error rate for lamda 10:", error_train_rate)
        print("Testing error rate for lamda 10:", error_test_rate)
    if (lamda == 100):
        print("Training error rate for lamda 100:", error_train_rate)
        print("Testing error rate for lamda 100:", error_test_rate)
    

#Plot training and test error rates vs list of lamda
plt.title("Q3. Logistic Regression")
plt.plot(lamda_list,error_train_l,'g',label ="Training",linewidth = 2)
plt.plot(lamda_list,error_test_l,'r',label ="Testing",linewidth = 2)
plt.xlabel("Lamda")
plt.ylabel("Error Rates")
plt.legend()
plt.grid(True)
plt.show()






