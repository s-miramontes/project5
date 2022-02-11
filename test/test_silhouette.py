# write your silhouette score unit tests here
import numpy as numpy
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
	assert np.all(get_scores >= -1) and np.all(get_scores <= 1)
	# based on ground truth
	ssert np.mean(get_scores) > 0.9

