"""
Tests for the DNA inputs: CTCF motif tracks (scripts/data/motif_tracks.py)
and language-model embeddings of enhancers and promoters
(scripts/data/dna_embeddings.py, src/dna_embeddings.py).
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "tests"))

import dna_embeddings  # noqa: E402
import motif_tracks  # noqa: E402
from src.dna_embeddings import ElementEmbeddings, element_key  # noqa: E402

CTCF = """>MA0139.2\tCTCF
A  [   281     56      8    744     40    107    851      5    333     54     12     56    104    372     82 ]
C  [    49    800    903     13    528    433     11      0      3     12      0      8    733     13    482 ]
G  [   449     21      0     65    334     48     32    903    566    504    890    775      5    507    307 ]
T  [   134     36      2     91     11    324     18      3      9    341      8     71     67     17     37 ]
"""
RC = str.maketrans("ACGT", "TGCA")


def test_motif_tracks_find_a_planted_site_on_each_strand():
    name, counts = motif_tracks.read_jaspar(CTCF)
    lod = motif_tracks.log_odds(counts)
    cut = motif_tracks.threshold(lod, 1e-5)
    best = "".join("ACGT"[i] for i in lod.argmax(axis=0))
    assert name == "CTCF" and 0 < cut < lod.max(axis=0).sum()
    seq = ["AT"] * 2500                               # 5,000 bp without a site
    seq = "".join(seq)
    seq = seq[:1240] + best + seq[1255:3240] + best.translate(RC)[::-1] + seq[3255:]
    plus, minus, matches = motif_tracks.motif_tracks({"chr1": seq}, lod, cut,
                                                     chroms={"chr1": len(seq)})
    assert matches == {"plus": 1, "minus": 1}
    assert len(plus["chr1"]) == 11                    # ceil(5000 / 500) + 1
    assert torch.nonzero(plus["chr1"]).flatten().tolist() == [2]
    assert torch.nonzero(minus["chr1"]).flatten().tolist() == [6]


def test_a_short_matrix_refuses_a_pvalue_it_cannot_reach():
    # JASPAR 2024's YY1 matrix (MA0095.4): an 8 bp core, whose best match
    # alone has p 4 ** -8 = 1.5e-5. At 1e-5 every position used to pass.
    yy1 = """>MA0095.4\tYy1
A  [  3123  41529  39666  30769  41506    689    561   1597 ]
C  [ 33122    926   1447   2536    746    241    165    971 ]
G  [  3684    287   1320   8658    712    708  42263  39612 ]
T  [  3419    606    915   1385    384  41710    359   1168 ]
"""
    name, counts = motif_tracks.read_jaspar(yy1)
    lod = motif_tracks.log_odds(counts)
    assert name == "Yy1"
    with pytest.raises(ValueError, match="best match has p"):
        motif_tracks.threshold(lod, 1e-5)
    assert 0 < motif_tracks.threshold(lod, 1e-4) < lod.max(axis=0).sum()


def test_embeddings_do_not_depend_on_the_strand():
    model = dna_embeddings.DummyModel()
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), size=600))
    forward = dna_embeddings.embed(model, [seq], 4)
    reverse = dna_embeddings.embed(model, [seq.translate(RC)[::-1]], 4)
    assert forward.shape == (1, 4, model.dim)
    assert np.allclose(forward, reverse[:, ::-1], atol=1e-6)


def write_genome(path, chroms, length=500_000):
    rng = np.random.default_rng(0)
    with open(path, "w") as handle:
        for chrom in chroms:
            bases = "".join(rng.choice(list("ACGT"), size=length))
            handle.write(f">{chrom}\n" + "\n".join(
                bases[i:i + 60] for i in range(0, length, 60)) + "\n")


def build_embeddings(bengi_dir, genome, out):
    script = str(ROOT / "scripts" / "data" / "dna_embeddings.py")
    common = ["--pairs", str(bengi_dir / "*.tsv*"), "--out", str(out),
              "--shards", "2", "--segments", "4", "--window", "300"]
    for shard in ("0", "1"):
        result = subprocess.run([sys.executable, script, *common, "--genome", str(genome),
                                 "--model", "dummy", "--shard", shard, "--device", "cpu"],
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, script, *common, "--merge"],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr


def test_extending_a_table_matches_embedding_everything(tmp_path):
    """--extend embeds only the missing elements; the merged table equals one
    built from all the files at once."""
    pytest.importorskip("pyfaidx")
    import shutil
    from test_end_to_end import CHROMS, write_dataset
    bengi_dir, _feats = write_dataset(tmp_path / "data")
    genome = tmp_path / "toy.fa"
    write_genome(genome, CHROMS)
    files = sorted(bengi_dir.glob("*.tsv*"))
    first = tmp_path / "first"
    first.mkdir()
    shutil.copy(files[0], first / files[0].name)
    build_embeddings(first, genome, tmp_path / "old")
    build_embeddings(bengi_dir, genome, tmp_path / "all")

    script = str(ROOT / "scripts" / "data" / "dna_embeddings.py")
    common = ["--pairs", str(bengi_dir / "*.tsv*"), "--out", str(tmp_path / "extended"),
              "--shards", "2", "--segments", "4", "--window", "300",
              "--extend", str(tmp_path / "old")]
    for shard in ("0", "1"):
        result = subprocess.run([sys.executable, script, *common, "--genome", str(genome),
                                 "--model", "dummy", "--shard", shard, "--device", "cpu"],
                                capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([sys.executable, script, *common, "--merge"],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr

    old, full = ElementEmbeddings(str(tmp_path / "old")), ElementEmbeddings(str(tmp_path / "all"))
    extended = ElementEmbeddings(str(tmp_path / "extended"))
    assert 0 < len(old.keys) < len(full.keys)
    assert np.array_equal(extended.keys, full.keys)
    assert np.array_equal(np.asarray(extended.array), np.asarray(full.array))
    assert not (tmp_path / "extended" / "new_elements.npy").exists()


def test_missing_embeddings_are_counted():
    class Table(ElementEmbeddings):
        def __init__(self, keys):
            self.keys = np.sort(np.asarray(keys, dtype=np.int64))
    known = element_key(["chr1", "chr1", "chr2"], [100, 200, 300])
    table = Table(known)
    assert table.count_missing(["chr1", "chr1"], [100, 100], [200, 999]) == 1
    assert table.count_missing(["chr2", "chrM"], [300, 100], [300, 200]) == 1


@pytest.mark.slow
def test_training_with_both_dna_inputs(tmp_path):
    pytest.importorskip("pyfaidx")
    from test_end_to_end import CELLS, CHROMS, run_training, write_config, write_dataset
    bengi_dir, feats = write_dataset(tmp_path / "data")
    genome = tmp_path / "toy.fa"
    write_genome(genome, CHROMS)
    out = tmp_path / "emb"
    build_embeddings(bengi_dir, genome, out)

    stored = ElementEmbeddings(str(out))
    assert stored.array.shape == (len(stored.keys), 4, 16)
    assert np.all(np.diff(stored.keys) > 0)
    chrom = f"chr{int(stored.keys[0] >> 32) + 1}"
    position = int(stored.keys[0] & 0xFFFFFFFF)
    assert element_key([chrom], [position])[0] == stored.keys[0]
    pair = stored.pair(chrom, position, 1)                 # no element at 1
    assert pair.shape == (8, 16) and stored.missing == 1
    assert torch.allclose(pair[:4], torch.from_numpy(stored.array[0].astype(np.float32)))
    assert torch.all(pair[4:] == 0)

    # Two genome-wide motif tracks, the same file for every cell.
    tracks = {c: torch.rand(4_000) for c in CHROMS}
    torch.save(tracks, tmp_path / "data" / "motif_plus.pt")
    torch.save(tracks, tmp_path / "data" / "motif_minus.pt")
    listed = json.loads(feats.read_text())
    for cell in CELLS:
        listed[cell]["CTCFmotif_plus"] = str(tmp_path / "data" / "motif_plus.pt")
        listed[cell]["CTCFmotif_minus"] = str(tmp_path / "data" / "motif_minus.pt")
    feats.write_text(json.dumps(listed))

    config_path = tmp_path / "config.yaml"
    write_config(config_path, bengi_dir, feats, epochs=1)
    config = yaml.safe_load(config_path.read_text())
    config["paths"].update(bengi_dirs=[str(bengi_dir)], dna_embeddings=str(out))
    config["data"].update(
        train_cells="all", modalities="all", dna_input="embedding", dna_segments=4,
        dna_embedding_dim=16, n_epigenetic_features=11,
        feats_order=config["data"]["feats_order"] + ["CTCFmotif_plus", "CTCFmotif_minus"])
    config["training"].update(valid_cells=["IMR90"], valid_samples=30, samples_per_epoch=64)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    for variant in ("KA", "baseline"):
        result = run_training(config_path, tmp_path / f"run_{variant}",
                              extra=("--variant", variant))
        assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
        assert "DNA input: language-model embeddings" in result.stdout
        assert "Loaded 10/10 marks" in result.stdout
    # Scoring a saved run reads the same inputs.
    scored = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_external.py"), "--runs",
         str(tmp_path / "run_KA"), "--tests", str(bengi_dir / "HMEC.HiC-Benchmark.v3.tsv"),
         "--feats-config", str(feats), "--abc-dir", str(tmp_path / "no_abc"),
         "--out", str(tmp_path / "external"), "--device", "cpu",
         "--dna-embeddings", str(out)],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert scored.returncode == 0, scored.stdout[-3000:] + scored.stderr[-3000:]
    assert (tmp_path / "external" / "summary.json").exists()
