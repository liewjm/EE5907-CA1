import scipy.io
import scipy.spatial.distance as dist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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

#================================FUNCTIONS=============================================
def KNN(dist,k):
    #Find K Nearest Neighbour of selected data sample - get idx of K smallest values of distance matrix 
    idx = np.argsort(dist)[:k] #idx of the K nearest neighbour 
    prob_label_0 = np.sum(ytrain.flatten()[idx]==0)/k
    prob_label_1 = np.sum(ytrain.flatten()[idx]==1)/k

    if (prob_label_1 >= prob_label_0):
        return 1
    return 0

#=================================MAIN=================================================
K_1 = np.arange(1,11)
K_2 = np.arange(15,105,5)
K_list = np.concatenate((K_1,K_2),axis = 0)
print(K_list)

error_train_l=[]
error_test_l=[]

#Create distance matrix for training-training data samples
train_dist = np.zeros((len(xtrain),len(xtrain)))
for i in tqdm(range(0,len(xtrain),1),desc = "Calculating training-training distance matrix"):
    datasample = xtrain[i]
    for j in range(0,len(xtrain),1):
        trainsample = xtrain[j]
        dist = (datasample - trainsample)**2 #Euclidean distance  
        train_dist[i][j]= np.sqrt(np.sum(dist))
print("Training-training distance matrix:",train_dist)

#Create distance matrix for testing-training data samples
test_dist = np.zeros((len(xtest),len(xtrain)))
for i in tqdm(range(0,len(xtest),1),desc = "Calculating testing-training distance matrix"):
    datasample_ = xtest[i]
    for j in range(0,len(xtrain),1):
        trainsample_ = xtrain[j]
        dist_ = (datasample_ - trainsample_)**2
        test_dist[i][j]= np.sqrt(np.sum(dist_))
print("Testin-training distance matrix:", test_dist,"\n")


#Predict target class of email data x using KNN 
for K in tqdm(K_list, desc="Predict y label for each data sample from training dataset"):
    #Predict y label for each data sample from training dataset 
    training_results = []
    for i in range(len(ytrain)):
        predict_y = KNN(train_dist[i],K)
        training_results.append(predict_y)

    #Predict y label for each data sample from testing dataset
    testing_results = [] 
    for i in range(len(ytest)):
        predict_y_test = KNN(test_dist[i],K)
        testing_results.append(predict_y_test)

    #Calculate training error - Compare predicted label vs true label for each data sample
    error_count_train= 0
    for idx in range(len(training_results)):
        if (training_results[idx]!= (ytrain.flatten())[idx]):
            error_count_train +=1
    error_train_rate = error_count_train/len(xtrain) 
    error_train_l.append(error_train_rate)

    #Calculate testing error - Compare predicted label vs true label for each data sample
    error_count_test = 0
    for idx in range(len(testing_results)):
        if (testing_results[idx]!= (ytest.flatten())[idx]):
            error_count_test +=1
    error_test_rate = error_count_test/len(xtest) 
    error_test_l.append(error_test_rate)
  
    if (K == 1):
        print("Training error rate for K 1:", error_train_rate)
        print("Testing error rate for K 1:", error_test_rate)
    if (K == 10):
        print("Training error rate for K 10:", error_train_rate)
        print("Testing error rate for K 10:", error_test_rate)
    if (K == 100):
        print("Training error rate for K 100:", error_train_rate)
        print("Testing error rate for K 100:", error_test_rate)


#Plot training and test error rates vs list of lamda
plt.title("Q4. K-Nearest Neighbours")
plt.plot(K_list,error_train_l,'g',label ="Training",linewidth = 2)
plt.plot(K_list,error_test_l,'r',label ="Testing",linewidth = 2)
plt.xlabel("K")
plt.ylabel("Error Rates")
plt.legend()
plt.grid(True)
plt.show()




