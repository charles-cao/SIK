import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

class SparseINNE:
    
    def __init__(self, max_samples=16, n_estimators=200, novelty=False, random_state=None):
        """
        Parameters
        ----------
        max_samples : int, default=16
            The number of samples to draw for each estimator.
        
        n_estimators : int, default=200
            The number of base estimators in the ensemble.
        
        novelty : bool, default=False
            If True, the model is used for novelty detection and must be fitted on a 
            dataset of normal samples. If False, the model is used for outlier detection 
            where training data may contain outliers.
        
        random_state : int, RandomState instance or None, default=None
            Controls the random seed used for generating random sample indices.
        """
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.novelty = novelty
        self.random_state = random_state
    
    def fit(self, data):
        """
        Fit the model using data as training samples.
        
        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            Training data.
            
        Returns
        -------
        self : object
            Returns self.
        """
        self.data = data
        n_samples, n_features = self.data.shape

        self.centroid = []
        self.centroids_radius = []
        
        random_state = check_random_state(self.random_state)
        self._seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            rnd = check_random_state(self._seeds[i])
            subIndex = rnd.choice(n_samples, self.max_samples, replace=False)
            self.centroid.append(subIndex)
            
            tdata = self.data[subIndex, :]
            
            tt_dis = cdist(tdata, tdata)
            radius = [] 
            for r_idx in range(self.max_samples):
                r = tt_dis[r_idx].copy()
                r[r < 0] = 0 
                r = np.delete(r, r_idx) 
                radius.append(np.min(r))
            self.centroids_radius.append(radius)
        
        
        # If not in novelty mode, compute KME
        if not self.novelty:
            # Calculate KME - this is a dense vector (one value per estimator)
            # Use sparse mean and convert only once at the end
            # Compute transformed data for training set using sparse matrix
            self.X_transformed_train_ = self._transform_internal(self.data)
            mean_values = self.X_transformed_train_.mean(axis=0)
            self.kme_ = np.asarray(mean_values).ravel()
        
        return self
    
    def _transform_internal(self, newdata):
        """
        Internal transform method using sparse matrices for efficiency.
        
        Returns a sparse matrix where each row represents a sample and
        each column represents an estimator. 1 indicates the sample is
        outside the hypersphere, 0 indicates it's inside.
        """
        n, _ = newdata.shape
        
        # Use COO format to collect coordinates and then create CSR directly
        rows = []
        cols = []
        
        for i in range(self.n_estimators):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            
            dis = cdist(tdata, newdata)
            
            centerIdx = np.argmin(dis, axis=0)
            
            min_distances = dis[centerIdx, np.arange(n)]
            
            radii = np.array(radius)[centerIdx]
            
            # Find indices where points are outside the hypersphere
            outside_indices = np.where(min_distances > radii)[0]
            
            # Collect coordinates for non-zero elements
            if len(outside_indices) > 0:
                rows.extend(outside_indices)
                cols.extend([i] * len(outside_indices))
        
        # Create data array (all 1s for binary case)
        data = np.ones(len(rows), dtype=np.int8)
        
        # Create CSR matrix directly
        result = csr_matrix((data, (rows, cols)), shape=(n, self.n_estimators), dtype=np.int8)
        
        return result
    
    def transform(self, X):
        """
        Transform X into the ensemble feature space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        X_transformed : scipy.sparse.csr_matrix of shape (n_samples, n_estimators)
            The transformed data. Each element is 0 if sample is inside the
            hypersphere, 1 if it's outside.
        """
        check_is_fitted(self, ["centroid", "centroids_radius"])
        
        # If novelty=False and X is the training data, return pre-computed result
        if not self.novelty and hasattr(self, 'X_transformed_train_') and np.array_equal(X, self.data):
            return self.X_transformed_train_
        
        return self._transform_internal(X)
    
    def decision_function(self, X):
        """
        Compute the anomaly score for each sample in X using the fitted model.
        Directly uses sparse matrix operations without unnecessary conversions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly scores of the input samples.
            Higher scores indicate higher likelihood of being anomalies.
        """
        check_is_fitted(self, ["centroid", "centroids_radius"])
        
        X_transformed = self.transform(X)
        
        if self.novelty:
            # For novelty detection, count non-zeros in each row (Hamming distance)
            # Keep the result as a sparse matrix until the end
            row_sums = X_transformed.sum(axis=1)
            # Convert only at the final step
            return np.asarray(row_sums).ravel()
        else:
            # For outlier detection with KME
            # The dot product result is already a dense array
            return X_transformed.dot(self.kme_)
