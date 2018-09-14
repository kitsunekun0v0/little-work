import pandas as pd
import numpy as np
from scipy.misc import logsumexp
from sklearn.decomposition import TruncatedSVD

train_data = pd.read_csv('trainsubset3000.csv',delimiter=',').values

#samples feature vectors & labels
sample_vector = train_data[:,1:-1]
sample_label = train_data[:,-1]

svd = TruncatedSVD(n_components=500, random_state=42)
svd.fit(sample_vector)
sample_vector = svd.transform(sample_vector)

##############train params################
classAll = np.unique(sample_label) # 30 classes
C = len(classAll)

#define params for gaussian naive bayes
# mean = 30x13626
#var = 30x13626
#p(C) = pi = 30x1
mean = np.zeros((C,len(sample_vector[0])))
var = np.zeros((C,len(sample_vector[0])))
pi = np.zeros(C)

#classAll = set(sample_label)
classFreq = np.zeros(C)


#calculating p(C=k) for each class
#count occurence of each class
index = 0
for c in classAll:
    for i in sample_label:
        if i==c:
            classFreq[index] +=1
    index+=1
#calculate pi
pi = classFreq/len(sample_label)


# estimating gaussian distribution params mean and var
#calculate mean
sumF = np.zeros((C,len(sample_vector[0])))
index = 0
for c in classAll:
    j = 0
    for i in sample_label:
        if i==c:
            sumF[index,:] = sumF[index,:] + sample_vector[j,:]
        j+=1
    index +=1


for i in range(0,C):
    mean[i,:] = sumF[i,:]/classFreq[i]


#calculate var
f_mean = np.zeros((C,len(sample_vector[0])))
index = 0
for c in classAll:
    j = 0
    for i in sample_label:
        if i==c:
            u = sample_vector[j,:] - mean[index,:]
            f_mean[index,:] = u*u + f_mean[index,:]
        j+=1
    index+=1

dv = 1 #dummy var
for i in range(0,C):
    var[i,:] = f_mean[i,:]/classFreq[i]
    for j in range(0,len(f_mean[0])):
        if var[i,j]==0:
            var[i,j] = var[i,j] + dv
            j+=1
        else:
            j+=1



################ make inference #################

test_data = pd.read_csv('testsubset.csv',delimiter=',').values

test_vector = test_data[:,1:-1]
test_label = test_data[:,-1]

test_vector = svd.transform(test_vector)

# predict labels for testing data
# p(y=k|x,D) proportional to p(y=k).product of each feature ~N(x*; mean, var)
"""""
compute log joint probability log p(x,y)
log p(x,y) = sum(log p(x|y))+log p(y)
"""""
predictLabel = []
for t in range(0,len(test_label)):
    p_C_list = np.zeros(C)  # list of log probability of input x and class c
    for i in range(0,C):
        p_C_list[i] = p_C_list[i] + np.log(pi[i]) # log p(y)
        p_C_list[i] = p_C_list[i] - 0.5 * np.sum(np.log(2. * np.pi * var[i,:]))
        p_C_list[i] = p_C_list[i] - 0.5 * (np.sum(((test_vector[t,:]-mean[i,:])**2)/(var[i,:]))+1)
    temp = list(classAll)[np.argsort(p_C_list)[-1]]
    predictLabel.append(temp)


count = 0
for i in range(0,len(test_label)):
    if predictLabel[i]==test_label[i]:
        count+=1

print(count)