import ray
import time
import sklearn.decomposition
import lda.datasets

import numpy as np
import scipy.sparse as sp
import warnings

from plasma import PlasmaClient
from scipy.special import gammaln
from sklearn.decomposition import online_lda
from sklearn.utils import _get_n_jobs

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import (check_random_state, check_array,
                     gen_batches, gen_even_slices, _get_n_jobs)
from sklearn.utils.validation import check_non_negative
from sklearn.utils.extmath import logsumexp
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals.six.moves import xrange
from sklearn.exceptions import NotFittedError
from scipy.sparse import csr_matrix

from sklearn.decomposition._online_lda import (mean_change, _dirichlet_expectation_1d,
                          _dirichlet_expectation_2d)

from multiprocessing.pool import ThreadPool

EPS = online_lda.EPS

NUM_WORKERS = 8
NUM_ITER = 100
# NUM_ITER = 10
addr_info = ray.init(start_ray_local=True, num_workers=NUM_WORKERS)

# SOURCE: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/decomposition/online_lda.py
def ray_update_doc_distribution(X, exp_topic_word_distr, doc_topic_prior,
                                max_iters,
                                mean_change_tol, cal_sstats, random_state):
    """E-step: update document-topic distribution.
    Parameters
    ----------
    X : array-like or sparse matrix, shape=(n_samples, n_features)
        Document word matrix.
    exp_topic_word_distr : dense matrix, shape=(n_topics, n_features)
        Exponential value of expection of log topic word distribution.
        In the literature, this is `exp(E[log(beta)])`.
    doc_topic_prior : float
        Prior of document topic distribution `theta`.
    max_iters : int
        Max number of iterations for updating document topic distribution in
        the E-step.
    mean_change_tol : float
        Stopping tolerance for updating document topic distribution in E-setp.
    cal_sstats : boolean
        Parameter that indicate to calculate sufficient statistics or not.
        Set `cal_sstats` to `True` when we need to run M-step.
    random_state : RandomState instance or None
        Parameter that indicate how to initialize document topic distribution.
        Set `random_state` to None will initialize document topic distribution
        to a constant number.
    Returns
    -------
    (doc_topic_distr, suff_stats) :
        `doc_topic_distr` is unnormalized topic distribution for each document.
        In the literature, this is `gamma`. we can calculate `E[log(theta)]`
        from it.
        `suff_stats` is expected sufficient statistics for the M-step.
            When `cal_sstats == False`, this will be None.
    """
    is_sparse_x = sp.issparse(X)
    n_samples, n_features = X.shape
    n_topics = exp_topic_word_distr.shape[0]

    if random_state:
        doc_topic_distr = random_state.gamma(100., 0.01, (n_samples, n_topics))
    else:
        doc_topic_distr = np.ones((n_samples, n_topics))

    # In the literature, this is `exp(E[log(theta)])`
    exp_doc_topic = np.exp(_dirichlet_expectation_2d(doc_topic_distr))

    # diff on `component_` (only calculate it when `cal_diff` is True)
    suff_stats = np.zeros(exp_topic_word_distr.shape) if cal_sstats else None

    if is_sparse_x:
        X_data = X.data
        X_indices = X.indices
        X_indptr = X.indptr

    for idx_d in xrange(n_samples):
        if is_sparse_x:
            ids = X_indices[X_indptr[idx_d]:X_indptr[idx_d + 1]]
            cnts = X_data[X_indptr[idx_d]:X_indptr[idx_d + 1]]
        else:
            ids = np.nonzero(X[idx_d, :])[0]
            cnts = X[idx_d, ids]

        doc_topic_d = doc_topic_distr[idx_d, :]
        # The next one is a copy, since the inner loop overwrites it.
        exp_doc_topic_d = exp_doc_topic[idx_d, :].copy()
        exp_topic_word_d = exp_topic_word_distr[:, ids]

        # Iterate between `doc_topic_d` and `norm_phi` until convergence
        for _ in xrange(0, max_iters):
            last_d = doc_topic_d

            # The optimal phi_{dwk} is proportional to
            # exp(E[log(theta_{dk})]) * exp(E[log(beta_{dw})]).
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS

            doc_topic_d = (exp_doc_topic_d *
                           np.dot(cnts / norm_phi, exp_topic_word_d.T))
            # Note: adds doc_topic_prior to doc_topic_d, in-place.
            _dirichlet_expectation_1d(doc_topic_d, doc_topic_prior,
                                      exp_doc_topic_d)

            if mean_change(last_d, doc_topic_d) < mean_change_tol:
                break
        doc_topic_distr[idx_d, :] = doc_topic_d

        # Contribution of document d to the expected sufficient
        # statistics for the M step.
        if cal_sstats:
            norm_phi = np.dot(exp_doc_topic_d, exp_topic_word_d) + EPS
            suff_stats[:, ids] += np.outer(exp_doc_topic_d, cnts / norm_phi)

    return (doc_topic_distr, suff_stats)

@ray.remote
def remote_e_step(X_id, W_id,
                  doc_topic_prior, max_doc_update_iter, mean_change_tol,
                  random_seed):
  random_state = np.random.RandomState(random_seed)
  X_local = X_id #ray.get(X_id)

  relevant_exp_dirichlet_component = W_id#ray.get(W_id)

  doc_topics, sstats = ray_update_doc_distribution(X_local,
                              relevant_exp_dirichlet_component,
                              doc_topic_prior,
                              max_doc_update_iter,
                              mean_change_tol,
                              cal_sstats=True,
                              random_state=random_state)

  # return [doc_topics, sstats]  TODO
  return sstats

def remote_em_step(model, X_ids, W_id, random_seed):
  suff_stats = np.zeros_like(model.components_)
  e_step_handles = [
    remote_e_step.remote(X_ids[i], W_id,
                         model.doc_topic_prior_, model.max_doc_update_iter,
                         model.mean_change_tol, random_seed)
    for i in range(NUM_WORKERS)]
  e_results = [ray.get(h) for h in e_step_handles]
  for sstats in e_results:
    suff_stats += sstats
  suff_stats *= model.exp_dirichlet_component_

  model.components_ = model.topic_word_prior_ + suff_stats
  model.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(model.components_))


X = lda.datasets.load_reuters().astype(np.float64)
n_samples, n_features = X.shape
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()
partitions = np.array_split(X, 8)
X_ids = [ray.put(i) for i in partitions]
# TODO
ShardedLDAModel = sklearn.decomposition.LatentDirichletAllocation

random_seed = int(time.time())
sharded_model = ShardedLDAModel(n_topics=20, n_jobs=8, max_iter=NUM_ITER,
                                learning_method='batch',
                                random_state=random_seed)
sharded_model._init_latent_vars(n_features)

W_id = ray.put(sharded_model.exp_dirichlet_component_)

times = []
for _ in xrange(10):
    global_t_start = time.time()
    for j in xrange(NUM_ITER):
      # t_start = time.time()

      remote_em_step(sharded_model, partitions, W_id, random_seed)
      W_id = sharded_model.exp_dirichlet_component_ #ray.put(sharded_model.exp_dirichlet_component_)
      # print("{}: duration {} s".format(i, time.time() - t_start))

    #   if j % 10 == 0:
    #     print("finished {} / {}".format(j, NUM_ITER))
      #
      #   doc_topics_distr, _ = sharded_model._e_step(X, cal_sstats=False,
      #                                               random_init=False,
      #                                               parallel=None)
      #   print("perplexity: {}".format(sharded_model.perplexity(X, doc_topics_distr, sub_sampling=False)))
    # print("ray: total duration {} s".format(time.time() - global_t_start))
    times.append(time.time() - global_t_start)

    doc_topics_distr, _ = sharded_model._e_step(X, cal_sstats=False,
                                                random_init=False,
                                                parallel=None)
    # print("final perplexity: {}".format(sharded_model.perplexity(X, doc_topics_distr)))
print times
