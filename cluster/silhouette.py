import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """

        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # placeholder for all scores
        scores = np.zeros(X.shape[0])

        # parwise dists for all points in X
        pair_dists = cdist(X, X, metric = self.metric)


        for idx in range(X.shape[0]):

            pair_dists = pair_dists[idx, y == y[idx]]

            # intracluster distance at idx where label y corresponds to label at idx
            intra_num = np.sum(pair_dists)
            intra_denom = np.sum(y == y[idx]) - 1
            intra = intra_num / intra_denom


            # intercluster arr to hold all dists
            inter_dists = np.ones(np.max(y)) * np.inf 

            for l in range(np.max(y)):

                if l != y[idx]:
                    inter_dists[l] = np.sum(pair_dists[idx, y == l])/np.sum(y == l)

            inter = np.min(inter_dists)
            scores[idx] = (inter - intra)/np.max([intra, inter])

        self.scores = scores

        return self.scores








