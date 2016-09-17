from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

###############################################

X, y = make_blobs(n_samples=200, #150,
                  n_features=2,
                  centers=6, #3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

plt.scatter( X[:,0],
             X[:,1],
             c='white',
             marker='o',
             s=50)

plt.grid()
plt.show()

################################################

# initial_centroids = [num_clusters, num_features]
initial_centroids = [[8,7],
                     [2,8]]
                     #[0,2]]
np_initial_centroids = np.array(initial_centroids,np.float64)

km = KMeans( n_clusters=2,
             init=np_initial_centroids, #'random',
             n_init=1, #10,
             max_iter=300,
             tol=1e-04,
             random_state=0)

y_km = km.fit_predict(X)
print y_km
rr=raw_input()

#################################################

plt.scatter(X[y_km==0,0],
            X[y_km==0,1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster1')
plt.scatter(X[y_km==1,0],
            X[y_km==1,1],
            s=50,
            c='orange',
            marker='o',
            label='cluster2')
plt.scatter(X[y_km==2,0],
            X[y_km==2,1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster3')
plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],
            s=250,
            marker='*',
            c='red',
            label='centroids')

plt.legend()
plt.grid()
plt.show()

################################################
print "<<<<<<DONE>>>>>>"
