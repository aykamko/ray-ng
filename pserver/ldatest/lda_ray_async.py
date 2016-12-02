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

def worker_client_initializer():
    return PlasmaClient(addr_info['object_store_name'])
ray.reusables.local_client = ray.Reusable(worker_client_initializer, lambda x: x)

def relevant_shard_ranges(X):
  # NOTE: assumes at least one element
  nzc = sorted(np.unique(np.nonzero(X)[1]))
  feat_ranges = []
  range_start = nzc[0]
  cur_range = [range_start, range_start+1]
  for nzidx in nzc[1:]:
    shard_idx = nzidx
    if shard_idx < cur_range[1]:
      continue
    elif shard_idx == cur_range[1]:
      cur_range[1] += 1
    else:
      feat_ranges.append(tuple(cur_range))
      cur_range = [shard_idx, shard_idx+1]
  feat_ranges.append(tuple(cur_range))
  return feat_ranges

@ray.remote(num_return_vals=NUM_ITER)
def async_e_step(X_id, W_id, X_shard_idx, X_shard_size, n_feats, W_shape,
                  doc_topic_prior, max_doc_update_iter, mean_change_tol):
  local_client = ray.reusables.local_client
  X_range = (X_shard_idx * X_shard_size, min((X_shard_idx+1)*X_shard_size, n_feats))
  X_local = local_client.pull(X_id, X_range)

  relevant_shards = relevant_shard_ranges(X_local)
  relevant_exp_dirichlet_component = np.zeros(W_shape)
  for i in range(NUM_ITER):
    # time.sleep(0.2)
    # TODO: collapse this to one lib call
    # relevant_exp_dirichlet_component.fill(0.0)
    # for shard_range in relevant_shards:
    #   relevant_exp_dirichlet_component[:, shard_range[0]:shard_range[1]] = \
    #     local_client.pull(W_id, shard_range)
    relevant_exp_dirichlet_component[:] = local_client.pull(W_id, (0, W_shape[-1]))

    doc_topics, sstats = ray_update_doc_distribution(X_local,
                                relevant_exp_dirichlet_component,
                                doc_topic_prior,
                                max_doc_update_iter,
                                mean_change_tol,
                                cal_sstats=True,
                                random_state=None) # TODO

    # return [doc_topics, sstats]  TODO
    yield sstats

def local_m_step(e_step_handles, model):
  suff_stats = np.zeros_like(model.components_)
  e_results = [ray.get(h) for h in e_step_handles]
  for sstats in e_results:
    suff_stats += sstats
  suff_stats *= model.exp_dirichlet_component_

  model.components_ = model.topic_word_prior_ + suff_stats
  model.exp_dirichlet_component_ = np.exp(_dirichlet_expectation_2d(model.components_))

def async_train(X, client, model, num_iters, X_id, W_id, X_shard_size, n_feats, W_shape):
  handle_matrix = np.empty((NUM_WORKERS, num_iters), dtype=object)
  for i in range(NUM_WORKERS):
    iter_handles = async_e_step.remote(X_id, W_id, i, X_shard_size, n_features, W.shape,
                                       sharded_model.doc_topic_prior_,
                                       sharded_model.max_doc_update_iter,
                                       sharded_model.mean_change_tol)
    handle_matrix[i, :] = iter_handles

  for j in xrange(num_iters):
    # t_start = time.time()

    local_m_step(handle_matrix[:, j], model)
    client.push(W_id, (0, W_shape[-1]), model.exp_dirichlet_component_, shard_order='F')

    # print("{}/{}: duration {} s".format(j+1, num_iters, time.time() - t_start))

    if j % 10 == 0:
      print("finished {} / {}".format(j, num_iters))
      # doc_topics_distr, _ = sharded_model._e_step(X, cal_sstats=False,
      #                                             random_init=False,
      #                                             parallel=None)
      # print("perplexity: {}".format(sharded_model.perplexity(X, doc_topics_distr)))


X = lda.datasets.load_reuters().astype(np.float64)
n_samples, n_features = X.shape
vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

client = PlasmaClient(addr_info['object_store_name'])

X_id = "x"*20
X_shard_size = n_samples / NUM_WORKERS
client.init_kvstore(X_id, X, shard_order='C', shard_size=X_shard_size)

# TODO
ShardedLDAModel = sklearn.decomposition.LatentDirichletAllocation

sharded_model = ShardedLDAModel(n_topics=20, n_jobs=8)
sharded_model._init_latent_vars(n_features)

W = sharded_model.exp_dirichlet_component_
W_id = "w"*20
W_SHARDS = 100
W_shard_size = n_features / W_SHARDS
client.init_kvstore(W_id, W, shard_order='F', shard_size=W_shard_size)

# NUM_ITER = 10
global_t_start = time.time()
async_train(X, client, sharded_model, NUM_ITER, X_id, W_id, X_shard_size, n_features, W.shape)
print("ray_async: total duration {} s".format(time.time() - global_t_start))

doc_topics_distr, _ = sharded_model._e_step(X, cal_sstats=False,
                                            random_init=False,
                                            parallel=None)
print("final perplexity: {}".format(sharded_model.perplexity(X, doc_topics_distr)))
