"""
Training-time measures against the locus shortcut.

The same enhancer and gene appear in many cell lines, often with the same
label. A model can then score a pair by recognising the locus rather than by
reading the cell's chromatin, and a distance-and-locus prior alone reaches
test AUPR 0.41 (scripts/audit_dataset.py). Three measures, each switched on in
the configuration and applied to the training rows only:

  class_balanced_negatives   data.negatives "class_balanced". For every
      positive, one negative with the same enhancer and one with the same
      gene, from the same cell line and of matched distance, in the spirit
      of CBOEP (Tsukada et al.), so enhancer and gene identity carry little
      label information.
  locus_prior_logits         training.debias "poe". A product of experts: a
      logistic model of distance and the leave-one-cell-out positive rates of
      the enhancer, the gene and the pair is fitted, and its logit is added
      to the network's in the training loss. The network is then rewarded
      only for what the prior cannot explain; validation and test use the
      network alone (Clark et al. 2019, He et al. 2019).
  discordant_pairs           training.rank_weight. Pairs of rows with the same
      enhancer and gene, positive in one training cell line and negative in
      another. A ranking loss asks the network to score the positive higher.
      Both inputs show the same locus, so only the chromatin can separate
      them.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.bengi import PairTable


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _dense(codes: np.ndarray) -> np.ndarray:
    """Codes renumbered 0..k-1."""
    return np.unique(codes, return_inverse=True)[1]


def leave_one_cell_out_rate(keys: np.ndarray, cells: np.ndarray, labels: np.ndarray,
                            strength: float = 5.0) -> np.ndarray:
    """Smoothed positive rate of each row's key, counted in the OTHER cells.

    The rate measures what a locus's label in other cell lines says about
    this row, without counting the row's own cell line. `strength` pulls
    rare keys toward the overall rate.
    """
    keys = _dense(keys)
    own = _dense(keys.astype(np.int64) * (int(cells.max()) + 1) + cells)
    y = labels.astype(np.float64)
    pos = np.bincount(keys, weights=y)[keys] - np.bincount(own, weights=y)[own]
    n = np.bincount(keys)[keys] - np.bincount(own)[own]
    base = y.mean()
    return (pos + strength * base) / (n + strength)


def prior_features(table: PairTable, index: np.ndarray) -> np.ndarray:
    """Distance and leave-one-cell-out locus rates of the rows at `index`."""
    c = table.columns
    labels = c["label"][index]
    cells = c["cell"][index]
    d = np.log10(np.maximum(c["dist"][index].astype(np.float64), 1.0))
    columns = [d, d ** 2]
    for keys in (c["enh"][index], c["gene"][index], table.pair_keys()[index]):
        columns.append(_logit(leave_one_cell_out_rate(keys, cells, labels)))
    return np.column_stack(columns)


def locus_prior_logits(table: PairTable, index, max_fit_rows: int = 1_000_000,
                       seed: int = 0) -> np.ndarray:
    """Logit of the distance-and-locus prior for each row at `index`."""
    from sklearn.linear_model import LogisticRegression
    index = np.asarray(index)
    x = prior_features(table, index)
    y = table.columns["label"][index]
    fit = np.arange(len(index))
    if len(fit) > max_fit_rows:
        fit = np.random.default_rng(seed).choice(fit, max_fit_rows, replace=False)
    mean, std = x[fit].mean(axis=0), x[fit].std(axis=0) + 1e-9
    model = LogisticRegression(max_iter=1000).fit((x[fit] - mean) / std, y[fit])
    return model.decision_function((x - mean) / std).astype(np.float32)


def class_balanced_negatives(table: PairTable, index, tolerance: float = 0.2,
                             seed: int = 0) -> np.ndarray:
    """The rows of `index` to keep: every positive, and for each positive up
    to one unused negative with the same enhancer and one with the same gene,
    from the same cell line, whose distance is within `tolerance` (relative)
    of the positive's. Returns positions in the table, sorted."""
    index = np.asarray(index)
    c = table.columns
    rng = np.random.default_rng(seed)
    labels, cells = c["label"][index], c["cell"][index]
    logd = np.log(np.maximum(c["dist"][index].astype(np.float64), 1.0))
    limit = np.log1p(tolerance)
    keep = [index[labels == 1]]
    for cell in np.unique(cells):
        mine = np.flatnonzero(cells == cell)
        positives = mine[labels[mine] == 1]
        negatives = mine[labels[mine] == 0]
        used = np.zeros(len(index), dtype=bool)
        chosen = []
        for key in ("enh", "gene"):
            groups = {}
            values = c[key][index[negatives]]
            for row, value in zip(negatives, values):
                groups.setdefault(int(value), []).append(int(row))
            for row in rng.permutation(positives):
                candidates = groups.get(int(c[key][index[row]]), ())
                best, gap = -1, limit
                for other in candidates:
                    g = abs(logd[other] - logd[row])
                    if not used[other] and g <= gap:
                        best, gap = other, g
                if best >= 0:
                    used[best] = True
                    chosen.append(best)
        keep.append(index[np.array(chosen, dtype=np.int64)])
    return np.sort(np.concatenate(keep))


def discordant_pairs(table: PairTable, index, seed: int = 0,
                     max_per_pair: int = 4) -> np.ndarray:
    """(positive, negative) row positions with the same enhancer and gene in
    two different cell lines. Each enhancer-gene pair gives up to
    `max_per_pair` such couples. Returns an (m, 2) array of table positions."""
    index = np.asarray(index)
    c = table.columns
    rng = np.random.default_rng(seed)
    keys = table.pair_keys()[index]
    labels = c["label"][index]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    out = []
    if not len(keys):
        return np.zeros((0, 2), dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    sorted_labels = labels[order]
    mixed = (np.maximum.reduceat(sorted_labels, starts)
             != np.minimum.reduceat(sorted_labels, starts))
    for a, b in zip(starts[mixed], ends[mixed]):
        rows = order[a:b]
        pos = rows[labels[rows] == 1]
        neg = rows[labels[rows] == 0]
        if not len(pos) or not len(neg):
            continue
        couples = [(p, n) for p in pos for n in neg
                   if c["cell"][index[p]] != c["cell"][index[n]]]
        if len(couples) > max_per_pair:
            couples = [couples[i] for i in rng.choice(len(couples), max_per_pair,
                                                      replace=False)]
        out += [(index[p], index[n]) for p, n in couples]
    return np.array(out, dtype=np.int64).reshape(-1, 2)
