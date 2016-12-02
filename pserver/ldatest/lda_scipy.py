import time
import lda.datasets
import sklearn.decomposition

import numpy as np

X = lda.datasets.load_reuters().astype(np.float64)
n_samples, n_features = X.shape
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

ShardedLDAModel = sklearn.decomposition.LatentDirichletAllocation

sharded_model = ShardedLDAModel(n_topics=20, n_jobs=8, max_iter=50, learning_method='batch')

t_start = time.time()
sharded_model.fit(X)
print("scipy: duration {} s".format(time.time() - t_start))
