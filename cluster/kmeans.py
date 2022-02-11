import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        # making sure the inputted k is greater than
        assert k > 0, "K must be greater than zero!"

        # making sure the num of iterations is +
        assert max_iter >0, "Num Iterations must be grater than zero!"

        # assert tol is slightly greater than zero
        assert tol > 0, "Tol must be sliiiightly greater than zero!"

        self.k = k 
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter


    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # get total num of examples features from mat
        self.num_samples, self.num_feats = mat.shape

        # initialize random centroids from mat
        self.centroids = self._random_centroids(mat)

        # starting with huge error to then minimize
        self._err = np.inf


        # keeping track of the number of iterations
        for _ in range(self.max_iter):

            # assign rest of points to centers in self.centroids
            clusters = self._cluster_points(mat)

            # old centroids vs new assigned based on mean
            prev_centroids = self.centroids 
            self.centroids = self._get_centroids(clusters, mat)

            # calculate mse, via cdist, np.square and np.mean - yay vector ops
            mse = np.average(np.square(np.min(cdist(mat, 
                                                self.centroids,
                                            metric = self.metric))))

            # delta errors, for checking diff.
            delta_err = self._err - mse

            if delta_err < self.tol:
                print("Clusters found -- Convergence Achieved)")
                self._err = mse
                break


    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """

        # get labels from the best saved centroids
        return self._cluster_points(mat)


    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """

        return self._err

    def get_centroids(self) -> np.ndarray: # new centroids?
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """

        return self.centroids

    def _random_centroids(self, mat: np.ndarray): # rename and change
        '''
        Private method to randomly generate centroids given k
        number of clusters, from the available points stored
        and available in the matrix 'mat'.
        '''

        # need a total of k (clusters) by total features centroids
        all_centroids = np.zeros((self.k, self.num_feats))

        # random selection of (k) P[oints]
        for p in range(self.k):
            # pick random point from all available
            aCentroid = mat[np.random.choice(range(self.num_examples))]
            # save it
            all_centroids[p] = aCentroid

        # return the randomly generated centroids
        return all_centroids


    def _cluster_points(self, mat: np.array):
        '''
        Private method to create clusters based on the proximity of
        each point. 
        Note that proximity is calculated based on the metric
        inputed -- in this case we take advantage of cdist, as
        imported above the start of this script.

        '''
        # get all distances first with the given self.metric
        dist_points = cdist(mat, self.centroids, metric = self.metric)

        # get points that are closest to the centroids (here labels are indices)
        k_clusters_min = np.argmin(dist_points, axis=1)

        return k_clusters_min

    
    def _get_centroids(self, clusters, mat):

        all_centroids = np.zeros((self.k, self.num_feats))

        for i, cluster in enumerate(clusters):
            update_centroid = np.mean(mat[cluster], axis=0)    
            all_centroids[i] = update_centroid

        return all_centroids



