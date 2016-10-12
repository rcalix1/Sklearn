from sklearn import decomposition
#import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn import datasets
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

###########################################################
## set parameters

np.set_printoptions(threshold=np.inf) #print all values in numpy array

####################################################################

f_numpy = open("data/12559TrainingDataset.csv",'r')
Matrix_data = np.loadtxt(f_numpy, delimiter=",", skiprows=1)

##################################################################

#A = trainX.shape[1]   #num features
#B = trainY.shape[1]   #num classes
#samples_in_train = trainX.shape[0]
#samples_in_test = testX.shape[0]
A = len(Matrix_data[0,:])  
print "num features,", A

#X=Matrix_data[:, [1,2,3,4,5,6]]
X = Matrix_data[:,:21] #[:,:149]
y = Matrix_data[:,150]


#print X[:20,:]
#print "reading done"
#rr = raw_input()
####################################################################
## used for tensorflow
## load data from csv

#trainX = np.genfromtxt("data/trainX_mnist.csv",delimiter=",",dtype=None)
#trainY = np.genfromtxt("data/trainY_mnist.csv",delimiter=",",dtype=None)
#testX = np.genfromtxt("data/testX_mnist.csv",delimiter=",",dtype=None)
#testY = np.genfromtxt("data/testY_mnist.csv",delimiter=",",dtype=None)


print "data has been loaded from csv"
############################################################
## percentage split
## create train and test sets, or put all data in train sets
## for k-fold cross validation
## also perform feature scaling
## 70% train and 30% test (hence 0.30)


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

## k-folds crodd validation all goes in train sets (hence 0.01)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)




############################################################
## feature scaling

sc = StandardScaler()


sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

###########################################################

X_std = sc.fit_transform(X)

###########################################################
# features (A) and classes (B)
#  A number of features, 784 in this example
#  B = number of classes, 10 numbers for mnist (0,1,2,3,4,5,6,7,8,9)
#A = trainX.shape[1]   #num features
#B = trainY.shape[1]   #num classes
#samples_in_train = trainX.shape[0]
#samples_in_test = testX.shape[0]
#A = len(X_train[0,:])  # Number of features
#B = len(y_train_onehot[0]) #num classes
#print "num features", A

############################################################
## PCA

pca = decomposition.PCA(n_components=2)

pca_codes = pca.fit_transform(X_std)
print pca_codes[1:20,:]

############################################################

pca_train_test = decomposition.PCA(n_components=2)
pca_train_test.fit(X_train_std)

pca_X_train_std =  pca_train_test.transform(X_train_std)
pca_X_test_std  = pca_train_test.transform(X_test_std)

############################################################
plt.scatter(pca_codes[y==0,0],
            pca_codes[y==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='no')
plt.scatter(pca_codes[y==1,0],
            pca_codes[y==1,1],
            s=50,
            c='orange',
            marker='o',
            label='yes')

plt.legend()
plt.grid()
plt.show()
############################################################

plt.scatter(pca_X_train_std[y_train==0,0],
            pca_X_train_std[y_train==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='no')
plt.scatter(pca_X_train_std[y_train==1,0],
            pca_X_train_std[y_train==1,1],
            s=50,
            c='orange',
            marker='o',
            label='yes')

plt.legend()
plt.grid()
plt.show()


############################################################

plt.scatter(pca_X_test_std[y_test==0,0],
            pca_X_test_std[y_test==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='no')
plt.scatter(pca_X_test_std[y_test==1,0],
            pca_X_test_std[y_test==1,1],
            s=50,
            c='orange',
            marker='o',
            label='yes')

plt.legend()
plt.grid()
plt.show()


############################################################

print "<<<<<<<<<DONE>>>>>>>>>>"
