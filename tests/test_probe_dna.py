"""Tests of the linear probe of the DNA embeddings (scripts/probe_dna.py), on
a small synthetic embedding table and synthetic rows."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import probe_dna  # noqa: E402
from src.bengi import PairTable  # noqa: E402
from src.dna_embeddings import ElementEmbeddings, element_key  # noqa: E402


def write_table(folder: Path, keys, array):
    folder.mkdir(parents=True)
    np.save(folder / "keys.npy", np.asarray(keys, dtype=np.int64))
    np.save(folder / "embeddings.npy", np.asarray(array, dtype=np.float16))
    (folder / "meta.json").write_text(json.dumps({"segments": array.shape[1],
                                                  "dim": array.shape[2]}))
    return ElementEmbeddings(str(folder))


def test_pooled_embeddings_are_part_means_and_flag_missing_elements(tmp_path):
    keys = np.sort(element_key(["chr1", "chr1", "chr2"], [100, 500, 100]))
    array = np.arange(3 * 2 * 4, dtype=np.float32).reshape(3, 2, 4)
    embeddings = write_table(tmp_path / "emb", keys, array)
    query = np.array([keys[2], keys[0], element_key(["chr3"], [7])[0], keys[2]])
    means, found = probe_dna.pooled(embeddings, query, chunk=1)
    assert found.tolist() == [True, True, False, True]
    np.testing.assert_allclose(means[0], array[2].mean(axis=0))
    np.testing.assert_allclose(means[1], array[0].mean(axis=0))
    assert not means[2].any()
    np.testing.assert_allclose(means[3], means[0])


def test_element_keys_follow_the_data_pipeline():
    rows = [{"label": 1, "dist": 900.0, "chrom": c, "enh_start": 1000, "enh_end": 1201,
             "tss_coord": 2000, "cell": "K562", "assay": "HiC", "enh_id": "e", "gene": "g"}
            for c in ("chr1", "chrM")]
    table = PairTable.from_rows(rows)
    enh, prom, known = probe_dna.element_keys(table, np.arange(2))
    # The enhancer midpoint and the gene start, as src/epi_data_pipeline.py
    # computes them and scripts/data/dna_embeddings.py keys them.
    assert enh[0] == element_key(["chr1"], [1100])[0]
    assert prom[0] == element_key(["chr1"], [2000])[0]
    assert known.tolist() == [True, False]


def test_only_the_product_set_sees_a_pairing_signal():
    rng = np.random.default_rng(0)
    n, d = 4000, 8
    enh, prom = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    log_dist = rng.uniform(3, 6, size=n)
    y = ((enh * prom).sum(axis=1) + 0.3 * rng.normal(size=n) > 1.0).astype(int)
    train, valid = slice(0, 3000), slice(3000, n)
    scores = {}
    for kind in probe_dna.SETS:
        x = probe_dna.features(kind, log_dist, enh, prom)
        assert x.shape[1] == {"distance": 1, "means": 1 + 2 * d, "product": 1 + 3 * d}[kind]
        scores[kind] = probe_dna.fit_and_score(x[train], y[train], x[valid], y[valid])["auroc"]
    assert scores["distance"] < 0.6 and scores["means"] < 0.6
    assert scores["product"] > 0.85
