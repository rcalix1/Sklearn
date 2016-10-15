import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


n_train = 20  # samples for training
n_test = 200  # samples for testing

n_features = 4


def generate_data(n_samples, n_features):
    """Generate random blob-ish data with noisy features.

    This returns an array of input data with shape (n_samples, n_features)
    and an array of n_samples target_labels

    Only one feature contains discriminative information, the other features
    contain only noise.
    """
    X, y = make_blobs(n_samples=n_samples, n_features=1, centers=[[-3],[3]])
   
    if n_features > 1:
        X = np.hstack([X, np.random.randn(n_samples, n_features - 1)])
    print X
    rr = raw_input()
    print y
    rr = raw_input()
    return X, y

################################################
## lda

X_train, y_train = generate_data(n_train, n_features)

plt.scatter(X_train[:,0],X_train[:,1])
plt.show()

clf1 = LinearDiscriminantAnalysis(solver='lsqr').fit(X_train, y_train)

X_test, y_test = generate_data(n_test, n_features)
score_clf1 = clf1.score(X_test, y_test)

print score_clf1

#########################################################
## pca

from sklearn import decomposition
from numpy import genfromtxt

###########################################################
## set parameters

#np.set_printoptions(threshold=np.inf) #print all values in numpy array

####################################################################

#f_numpy = open("data/12559TrainingDataset.csv",'r')
#Matrix_data = np.loadtxt(f_numpy, delimiter=",", skiprows=1)

############################################################
## PCA

pca = decomposition.PCA(n_components=2)

pca_codes = pca.fit_transform(X_train)
print pca_codes[:,:]

############################################################

pca = decomposition.PCA(n_components=2)
pca.fit(X_train)

pca_X_train =  pca.transform(X_train)
pca_X_test  = pca.transform(X_test)

############################################################
plt.scatter(pca_codes[y_train==0,0],
            pca_codes[y_train==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='no')
plt.scatter(pca_codes[y_train==1,0],
            pca_codes[y_train==1,1],
            s=50,
            c='orange',
            marker='o',
            label='yes')

plt.legend()
plt.grid()
plt.show()
############################################################


