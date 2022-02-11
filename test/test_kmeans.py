import numpy as np 
import pytest

# import this way to use sklearn's
import cluster

# there we go
import sklearn
from sklearn.cluster import KMeans

def test_inputs():
	"""
	Testing inputs to KMeans are correct
	"""

	with pytest.raises(AssertionError, match = "K must be greater than zero!"):
		kms = cluster.KMeans(k = 0)

	with pytest.raises(AssertionError, match = "Num Iterations must be grater than zero!"):
		kms = cluster.KMeans(k = 2, max_iter = -1)

	with pytest.raises(AssertionError, match = "Tol must be sliiiightly greater than zero!"):
		kms = cluster.KMeans(k = 2, tol = 0)


def test_construction():
	"""
	Testing we have the expected attributes after
	generating data (e.g. correct number of clusters,
	observations, features).
	"""

	# make data
	clusters, labels = cluster.make_clusters(n=100, m=10, k=13, scale=1.3)

	# calling implemented KMeans, and fitting
	kmeans = cluster.KMeans(k=13)
	kmeans.fit(clusters)

	# asserting purely based on input
	assert kmeans.num_samples == 100
	assert kmeans.num_feats == 10

	# do we have 13 centroids?
	assert len(kmeans.centroids) == 13

	# is the error close to zero?
	#assert np.isclose(kmeans._err, 0)

	print("THIS IS THE ERROR", kmeans._err)


def test_kmeans():
   """ 
   Testing KMeans implementation with generated
   data in utiils, here we test against the clusters
   generated in sklearn.
   """

   # super simple case, everyone has to cluster to 1
   clust, label = cluster.make_clusters(k =1)
   kmeans = cluster.KMeans(k=1)
   kmeans.fit(clust)

   y_preds = kmeans.predict(clust)
   # there is only one label (so it should be 0)
   unique = np.unique(y_preds)

   # is this the right call?
   assert unique == 0


   # testing something a bit more complext
   data = np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91]])

   #initializing both
   kmeans_silvia = cluster.KMeans(k=2)
   # vs
   kmeans_sklearn = KMeans(n_clusters=2)

   # fitting both
   kmeans_silvia.fit(data)
   kmeans_sklearn.fit(data)

   sk_pred = kmeans_sklearn.labels_
   silvia_pred = kmeans_silvia.predict(data)

   assert sk_pred.all() == silvia_pred.all()