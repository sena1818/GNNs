# usage: see docstring of class DensityForest

import numpy as np
from joblib import Parallel, delayed

class Node:
    pass


class DensityTree:
    def __init__(self, n_min=10):
        ''' n_min: minimum required number of instances in leaf nodes
        '''
        self.n_min = n_min
        self.bins = {
            'count': [],
            'mean': [],
            'cov': []
        }

    def fit(self, features, D_try=None):
        '''
        features: the feature matrix of the training sets
        '''
        N, D = features.shape

        if D_try is None:
            D_try = int(np.sqrt(D)) # number of features to consider for each split decision

        # initialize the root node
        self.root = Node()
        self.root.features  = features

        # build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            active_indices = self.select_active_indices(D, D_try)
            left, right = self.make_split_node(node, active_indices)
            if left is None: # no split found
                self.make_leaf_node(node)
            else:
                stack.append(left)
                stack.append(right)

        self.weights_ = np.array(self.bins['count'])
        self.weights_ = self.weights_ / self.weights_.sum()
        self.means_ = np.array(self.bins['mean'])
        self.covariances_ = np.array(self.bins['cov'])
        self.n_components = self.weights_.shape[0]

    def make_split_node(self, node, indices):
        '''
        node: the node to be split
        indices: a numpy array of length 'D_try', containing the feature
                         indices to be considered for the present split

        return: None, None -- if no suitable split has been found, or
                left, right -- the children of the split
        '''

        # find best feature j_min (among 'indices') and best threshold t_min for the split
        l_min = float('inf')  # upper bound for the loss, later the loss of the best split
        j_min, t_min = None, None

        for j in indices:
            thresholds = self.find_thresholds(node, j)

            # compute loss for each threshold
            for t in thresholds:
                loss = self.compute_loss_for_split(node, j, t)

                # remember the best split so far
                # (the condition is never True when loss = float('inf') )
                if loss < l_min:
                    l_min = loss
                    j_min = j
                    t_min = t

        if j_min is None: # no split found
            return None, None

        # create children for the best split
        left, right = self.make_children(node, j_min, t_min)

        # turn the current 'node' into a split node
        # (store children and split condition)
        node.left = left
        node.right = right
        node.split_index = j_min
        node.threshold = t_min

        return left, right

    def select_active_indices(self, D, D_try):
        ''' return a 1-D array with D_try randomly selected indices from 0...(D-1).
        '''
        return np.random.choice(range(D), size=D_try, replace=False)

    def find_thresholds(self, node, j):
        ''' return: a 1-D array with all possible thresholds along feature j
        '''
        if node.features.shape[0] / 2 < self.n_min:
            return []
        sort = np.sort(node.features[:,j])
        t = (sort[:-1] + sort[1:]) / 2
        if self.n_min > 1:
            t = t[(self.n_min-1): (-self.n_min+1)]
        return t

    def make_children(self, node, j, t):
        ''' execute the split in feature j at threshold t

            return: left, right -- the children of the split, with features
                                   properly assigned according to the split
        '''
        left = Node()
        right = Node()

        left.features = node.features[node.features[:,j] <= t]
        right.features = node.features[node.features[:,j] > t]

        return left, right

    def entropy_gaussian(self, features):
        ''' calculate the differential entropy of a d-variate Gaussian density

            features: a numpy array
            return: entropy
        '''
        if features.shape[0] == 1:
            print(features)
        sigma = np.linalg.det(np.cov(features.T))
        entropy = np.multiply(np.power((2 * np.pi * np.e), features.shape[-1]), sigma)
        if entropy <= 0:
            return 0
        entropy = np.log(entropy) / 2
        if np.isnan(entropy):
            entropy = np.infty
        return entropy


    def compute_loss_for_split(self, node, j, t):
        # return the loss if we would split the instance along feature j at threshold t
        # or float('inf') if there is no feasible split
        l_response = node.features[node.features[:,j] <= t]
        r_response = node.features[node.features[:,j] > t]
        if len(l_response) <= 1 or len(r_response) <= 1:
            return float('inf')

        loss = self.entropy_gaussian(l_response) + self.entropy_gaussian(r_response)
        return loss

    def make_leaf_node(self, node):
        # turn node into a leaf node
        self.bins['count'].append(node.features.shape[0])
        self.bins['mean'].append(node.features.mean(0))
        self.bins['cov'].append(np.cov(node.features.T) + 1e-6 * np.eye(node.features.shape[1]))

    def sample(self, n_samples):
        samples = []

        # Choose the components of samples
        component_choices = np.random.choice(self.n_components, size=n_samples, p=self.weights_)

        for k in range(self.n_components):
            # Compute the numbers of samples of the component k
            n_k = np.sum(component_choices == k)
            if n_k > 0:
                # Generate the samples from the normal distribution with the mean and the covariance matrix of component k
                samples_k = np.random.multivariate_normal(mean=self.means_[k],
                                                          cov=self.covariances_[k],
                                                          size=n_k)
                samples.append(samples_k)

        # Combine samples
        samples = np.vstack(samples)

        # Shuffle samples
        np.random.shuffle(samples)

        return samples

def bootstrap_sampling(features):
    '''return a bootstrap sample of features
    '''
    N = features.shape[0]
    inds = np.random.choice(N, N, replace=True)
    return features[inds]

class DensityForest():
    '''
    This class implements a random forest for density estimation, i.e. a
    collection of density trees according to the method described in
    Criminisi, Shotton & Konukoglu (2011): Decision Forests, Chapter 5
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/CriminisiForests_FoundTrends_2011.pdf

    Each tree performs a recursive subdivision of the training domain
    into adaptive bins. For the trees to differ from one another, this
    is randomized in the usual way (i.e. each tree only sees a bootstrap
    sample of the training set, and only a random subset of the features
    is considered in each split decision).
    The bins of the trees contain a Gaussian fit to the instances within the
    bin -- this works much better than a uniform distribution over the bin.

    To sample from the forest, first a tree is selected uniformly at random.
    Then a bin in that tree is selected according to the probability of each
    bin, i.e. using the discrete distribution over the fraction of instances
    located in each bin. Finally, a sample is generated from the Gaussian in
    the selected bin.

    Usage:

    from density_forest import DensityForest

    Xtrain = ...  # load/create the training set
    min_samples_per_bin = 5

    model = DensityForest(n_min=min_samples_per_bin)
    model.fit(Xtrain)

    samples = model.sample(n_samples=20)
    '''
    def __init__(self, n_trees=10, n_min=1):
        self.n_trees = n_trees
        self.trees = [DensityTree(n_min) for _ in range(self.n_trees)]

    def fit(self, features):
        self.trees = Parallel(n_jobs=-1)(delayed(self.fit_tree)(tree, features) for tree in self.trees)

    def fit_tree(self, tree, features):
        bootstrap_features = bootstrap_sampling(features)
        tree.fit(bootstrap_features)
        return tree

    def sample(self, n_samples):
        samples = []

        # Choose the trees of samples
        tree_choices = np.random.choice(self.n_trees, size=n_samples)

        for k in range(self.n_trees):
            # Compute the numbers of samples of the tree k
            n_k = np.sum(tree_choices == k)
            if n_k > 0:
                # Generate the samples from the tree k
                samples_k = self.trees[k].sample(n_k)
                samples.append(samples_k)

        # Combine samples
        samples = np.vstack(samples)

        # Shuffle samples
        np.random.shuffle(samples)

        return samples