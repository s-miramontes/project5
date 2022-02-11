'''
Implementation of KMeans and Slihouette score
'''

from .kmeans import KMeans
from .silhouette import Silhouette
from .utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

__version__ = '0.0.1'