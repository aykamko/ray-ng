import time
import sklearn.decomposition
import lda
import lda.datasets

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln
import warnings

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

X = lda.datasets.load_reuters()

vocab = lda.datasets.load_reuters_vocab()
titles = lda.datasets.load_reuters_titles()

EPS = online_lda.EPS

def _update_doc_distribution(X, exp_topic_word_distr, doc_topic_prior,
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

class ShardedLDAModel(sklearn.decomposition.LatentDirichletAllocation):
  def extract_relevant_components(self, X_split):
    W_split = [np.zeros(self.exp_dirichlet_component_.shape) for _ in range(len(X_split))]
    relavent_feats = [np.unique(np.nonzero(x)[1]) for x in X_split]
    for W, f in zip(W_split, relavent_feats):
      W[:, f] = self.exp_dirichlet_component_[:, f]
    return W_split

  # Modified from: https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/decomposition/online_lda.py#L319
  def sharded_e_step(self, X, cal_sstats, random_init, parallel=None):
    random_state = self.random_state_ if random_init else None

    X_split = np.array_split(X, self.n_jobs)
    exp_dirichlet_component_split = self.extract_relevant_components(X_split)

    n_jobs = _get_n_jobs(self.n_jobs)
    if parallel is None:
      parallel = Parallel(n_jobs=n_jobs, verbose=max(0, self.verbose - 1))
    results = parallel(
      delayed(_update_doc_distribution)(X_split[job_idx],
                                        exp_dirichlet_component_split[job_idx],
                                        self.doc_topic_prior_,
                                        self.max_doc_update_iter,
                                        self.mean_change_tol,
                                        cal_sstats,
                                        random_state)
      for job_idx in range(n_jobs))

    # merge result
    doc_topics, sstats_list = zip(*results)
    doc_topic_distr = np.vstack(doc_topics)

    if cal_sstats:
      # This step finishes computing the sufficient statistics for the
      # M-step.
      suff_stats = np.zeros(self.components_.shape)
      for sstats in sstats_list:
        suff_stats += sstats
        suff_stats *= self.exp_dirichlet_component_
    else:
      suff_stats = None

    return (doc_topic_distr, suff_stats)

  def _e_step(self, *args, **kwargs):
    return self.sharded_e_step(*args, **kwargs)



sharded_model = ShardedLDAModel(n_topics=20, n_jobs=8)
t_start = time.time()
sharded_model.fit(X)
t_dur = time.time() - t_start
print('Training duration: {} seconds'.format(t_dur))
print()

topic_word = sharded_model.components_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))



sklearn_model = sklearn.decomposition.LatentDirichletAllocation(n_topics=20, n_jobs=8)
t_start = time.time()
sklearn_model.fit(X)
t_dur = time.time() - t_start
print('Training duration: {} seconds'.format(t_dur))
print()

topic_word = sklearn_model.components_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
