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
        # not entirely sure I need this
        self.k = k 
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter

        # you shouldn't initialize a matrix mat here since
        # you are simply initializing the model, as in saying
        # how many clusters, what metric, tolerance, and
        # total number of iterations.
        # the fitting of data comes afterward, and by generating
        # data to THEN fit. That is, you pass it in to the fit
        # function. 
        # so you treat mat abstractly. 

    def _random_centroids(self, mat: np.ndarray):
        '''
        Private method to randomly generate centroids given k
        number of clusters, from the available points stored
        in mat.
        '''

        # need a total of k by total features centroids
        all_centroids = np.zeros((self.k, self.num_feats))

        # random selection of (k) P[oints]
        for p in range(self.k):
            # pick random point from all available
            aCentroid = np.random.choice(range(self.num_examples))
            # save it
            all_centroids[p] = all_centroids

        # return the randomly generated centroids
        return all_centroids

    def _get_neighbors(self, mat, centroids, metric):
        '''
        Private method to create clusters based on the proximity of
        each point. 
        Note that proximity is calculated based on the metric
        inputed. 
        '''

        # need k number of clusters
        k_clusters = [[] for _ in range(self.k)]

        # calculate distance to centroid based on given metric
        # I guess this follows the sklearn API
        assert metric == 'euclidean', "Must include euclidean metric."

        for j, x in enumerate(mat):
            # punny
            nearest_neighbor = np.argmin(np.sqrt(np.sum(x - centroids) ** 2,
                                         axis = 1))
            k_clusters[nearest_neighbor].append(j)

        return k_clusters
    

    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # get num examples, and num features
        self.num_examples, self.num_feats = mat.shape

        # steps:
        # generate random centroids based on mat


        # populate other poitns according to the clonseness to centroids above
        # given the number of iterations

        # how to decide on old centroid vs new one?
        # is this where the tolerance is used?

        # when to predict?
        # what about order?




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

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
