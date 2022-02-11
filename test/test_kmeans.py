import numpy as np 
import pytest

# import this way to use sklearn's
import cluster

# there we go
from sklearn.cluster import KMeans

def test_inputs():
	"""
	Testing inputs to KMeans are correct
	"""

	with pytest.raises(AssertionError, match = "K must be greater than zero!")
	with pytest.raises(AssertionError, match = "Num Iterations must be grater than zero!")
	with pytest.raises(AssertionError, match = "Tolerance must be greater than zero!")


def test_construction():
	"""
	Testing we have the expected attributes after
	generating data (e.g. correct number of clusters,
	observations, features).
	"""

	# make data
	clusters, labels = make_clusters(n=100, m=10, k=13, scale=1.3)

	# calling implemented KMeans, and fitting
	kmeans = cluster.KMeans(k=13)
	kmeans.fit(clusters)

	# asserting purely based on input
	assert kmeans.num_samples == 100
	assert kmeans.num_feats == 10

	# do we have 13 centroids?
	assert kmeans.centroids == 13

	# is the error close to zero?
	assert np.isclose(kmeans.mse, 0)


def test_kmeans():
   """ 
   Testing KMeans implementation with generated
   data in utiils, here we test against the clusters
   generated in sklearn.
   """

   # super simple case, everyone has to cluster to 1
   clust, label = make_clusters(k =1)
   kmeans = cluster.KMeans(k=1)
   kmeans.fit(clust)

   y_preds = kmeans.predict(clust)
   # there is only one label (so it should be 0)
   unique = np,unique(y_preds)

   # is this the right call?
   assert unique == 0

   # testing a large k
   c, l = make_clusters(k=60, scale=0.2)
   kmeansBig = cluster.KMeans(k=60)
   kmeansBig.fit(c)
   labels_big = kmeansBig.predict(c)

   # assert there is 60 labels
   assert len(np.unique(labels_big)) == 60

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
   kmeans_siliva.fit(data)
   kmeans_sklearn.fit(data)

   sk_pred = kmeans_sklearn.labels_
   silvia_pred = kmeans_silvia.predict(data)

   assert sk_pred.all() == silvia_pred.all()