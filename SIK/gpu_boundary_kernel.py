import numpy as np
from random import sample
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from sklearn.utils.validation import check_is_fitted, check_random_state

MAX_INT = np.iinfo(np.int32).max
MIN_FLOAT = np.finfo(float).eps

# 检测是否有PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class GPU_BK_INNE:
    
    def __init__(self, max_samples=16, n_estimators=200, novelty=False, random_state=None, use_gpu=True):
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
            
        use_gpu : bool, default=True
            Whether to use GPU acceleration if available.
        """
        self.max_samples = max_samples
        self.n_estimators = n_estimators
        self.novelty = novelty
        self.random_state = random_state
        self.use_gpu = use_gpu and HAS_TORCH
        
        # 设置计算设备
        self.device = None
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                print("警告: 未检测到GPU设备，将使用CPU计算。")
        else:
            self.device = torch.device("cpu") if HAS_TORCH else None
    
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
        self.data = np.asarray(data, dtype=np.float32)
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
            self.X_transformed_train_ = self._transform_internal(self.data)
            self.kme_ = np.average(self.X_transformed_train_, axis=0)
        
        return self
    
    def _transform_internal(self, newdata):
        """Internal transform method used by both fit and transform."""
        n, _ = newdata.shape
        
        result = np.zeros((n, self.n_estimators))
        
        for i in range(self.n_estimators):
            subIndex = self.centroid[i]
            radius = self.centroids_radius[i]
            tdata = self.data[subIndex, :]
            
            if self.use_gpu and self.device is not None:
                # 使用GPU加速距离计算
                tdata_tensor = torch.tensor(tdata, device=self.device)
                newdata_tensor = torch.tensor(newdata, device=self.device)
                dis = torch.cdist(tdata_tensor, newdata_tensor).cpu().numpy()
            else:
                # 使用原始的cdist
                dis = cdist(tdata, newdata)
            
            centerIdx = np.argmin(dis, axis=0)
            
            min_distances = dis[centerIdx, np.arange(n)]
            
            radii = np.array(radius)[centerIdx]
            
            # in ball: 0; out ball: 1
            result[:, i] = (min_distances > radii).astype(int)
        
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
        X_transformed : ndarray of shape (n_samples, n_estimators)
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
            # For novelty detection (training data is all normal)
            # Count how many estimators classify the point as outside the hypersphere
            # This is equivalent to Hamming distance from origin (all zeros)
            scores = np.sum(X_transformed, axis=1)
        else:
            # For outlier detection, use the KME approach
            scores = np.dot(X_transformed, self.kme_.T)
            # scores = np.sum(X_transformed, axis=1)
        
        return scores
