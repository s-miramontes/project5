# write your silhouette score unit tests here
import numpy as np
from cluster import(Silhouette, make_clusters)

# import sklearn silhouettes?
# from sklearn.metrics import sil...

def test_silhouette():
	"""
	Testing silhouette score, using make_clusters
	and corresponding labels to test error
	bounds.
	"""

	clusters, labels = make_clusters(scale=0.2, n=500)

	# start
	ss = Silhouette()

	# get scores for each clabel in clusters
	get_scores =ss.score(clusters, labels)

	# check bounds
	assert get_scores.all() >= -1 and get_scores.all() <= 1


