# SLIM

Cross-cell-line prediction of enhancer-promoter interactions with a
survival-gated memory encoder.

An enhancer-promoter interaction, written here as EPI, is a physical contact
between a distal regulatory element and a gene promoter. Predicting these
contacts in a cell type the model has never seen is the hard version of the
problem, because the DNA sequence is shared between cell types while the
chromatin state is not.

This repository holds everything needed to train the models, reproduce every
number in the manuscript, and run the analyses that support it.

## The method in one paragraph

Two branches read each candidate pair. The sequence branch encodes a 6,000
base-pair window of DNA around the enhancer and the promoter, using a
position-aware trinucleotide encoding followed by convolution and a
bidirectional recurrent layer, producing 128 tokens. The chromatin branch reads
eight experimental tracks plus a position channel across a 2.5 megabase window
in 500 base-pair bins, producing 500 tokens. The 628 tokens then pass through
the encoder that this work contributes.

Instead of letting every token attend to every other token, each encoder layer
does three things. Local windowed attention mixes each token with its 64
nearest neighbours. A survival gate then scores every token and writes the
eight highest-scoring ones into a bounded memory of 16 slots, through a gated
recurrent update with decay. Finally every token reads that memory back by
cross-attention. Distant information therefore travels through a small, explicit
memory rather than through all-to-all attention, and the selection step is
learned rather than fixed.

The survival score has three parts: a learned term, a novelty term measuring how
unlike the current memory a token is, and a prediction-error term measuring how
poorly the memory predicts that token. Selection is discrete in the forward pass
and differentiable in the backward pass through a straight-through estimator.

## Models

Four models share the branches, the pooling, the prediction heads, the data
pipeline, the optimizer and the training budget. The encoder is the only thing
that changes, and among the three SLIM variants only the feed-forward
sublayer inside the encoder changes.

| Name     | Encoder                       | Feed-forward sublayer            | Parameters |
| -------- | ----------------------------- | -------------------------------- | ---------: |
| baseline | global self-attention         | spline (Kolmogorov-Arnold) layer |  3,529,796 |
| A        | survival-gated memory encoder | rectified layer, width 720       |  4,340,255 |
| KA       | survival-gated memory encoder | spline layer, width 64           |  4,251,155 |
| GA       | survival-gated memory encoder | gated linear layer, width 480    |  4,340,975 |

Two comparisons matter, and each isolates one change:

- **baseline against KA** isolates the attention mechanism. Both carry the
  spline feed-forward layer, so the only difference is global self-attention
  against the survival-gated memory encoder. This carries the central claim.
- **A against KA against GA** isolates the feed-forward sublayer, since
  everything outside it is shared.

The rectified and gated layers are sized to hold the same number of parameters,
following the usual convention of scaling a gated unit's width by two thirds.
Their comparison therefore measures the layer type rather than its capacity.
The spline layer is 11 percent smaller; that difference is reported rather than
hidden.

All four models live in one class, `src/slim_model.SLIM`, with the
feed-forward sublayer selected by configuration. `tests/test_variants_matched.py`
asserts that the variants agree everywhere outside that sublayer, so a future
edit that touches one variant and not the others fails the test suite rather
than quietly invalidating an ablation.

## Repository layout

```
run.py            runs every stage in order, resuming after interruptions
configs/          base.yaml plus one thin overlay per model
src/              library code
  config.py            configuration loading with inheritance
  epi_data_pipeline.py reads BENGI pairs and binned chromatin tracks
  dataset.py           PyTorch dataset, augmentation, sequence encoding
  encoding.py          the position-aware trinucleotide encoder
  model_layers.py      spline layers, structured pooling, positional encoding
  memory_encoder.py      the survival-gated memory encoder
  slim_model.py     the two-branch model, all variants
  baseline_model.py    the global self-attention baseline
  metrics.py           metrics, threshold selection, calibration, bootstrap
  interpretation.py    gradient-weighted activation maps
  visualize.py         loss curves
scripts/          entry points, one per job
  train.py             training, with per-epoch resume
  evaluate.py          thresholds, calibration, intervals, leakage subsets
  audit_leakage.py     genomic overlap between the splits
  check_overfitting.py the overfitting gap and the transfer gap
  aggregate_seeds.py   per-seed values, means, paired differences
  benchmark_efficiency.py  latency, throughput, peak memory
  memory_writelog.py   what the survival gate selects
  build_results.py     tables and figures from saved predictions
results/          one directory per model, holding per-example predictions
tests/            run these before any long job
```

## Hardware and installation

Developed and run on a single NVIDIA RTX 5070 Ti with 16 GB of video memory and
32 GB of system memory. A batch size of 64 fits with roughly 2.5 GB to spare on
the heaviest variant. Batch 128 exhausts memory.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install PyTorch with the CUDA build that matches your driver, from
<https://pytorch.org/get-started/locally/>. The rest installs from PyPI.

Check the installation without any data:

```bash
python run.py --check              # packages, GPU, disk, data files
python -m pytest -q                # 25 tests, about 10 seconds
python -m pytest -q -m slow        # end to end on synthetic data, about 30 seconds
```

The slow test builds a miniature dataset in the real BENGI and track formats,
runs `scripts/train.py` on it as a subprocess, interrupts it, resumes it, and
then runs the analysis scripts over the outputs. If it passes, the pipeline is
sound whatever the real data does.

## Data

Three inputs are needed. None is redistributed here.

**1. BENGI benchmark pairs.** Clone <https://github.com/weng-lab/BENGI> and copy
the natural-ratio benchmark files into `data/BENGI/`. Six cell lines are used:

```
data/BENGI/GM12878.HiC-Benchmark.v3.tsv.gz
data/BENGI/HeLa.HiC-Benchmark.v3.tsv.gz
data/BENGI/K562.HiC-Benchmark.v3.tsv.gz
data/BENGI/IMR90.HiC-Benchmark.v3.tsv.gz
data/BENGI/HMEC.HiC-Benchmark.v3.tsv.gz
data/BENGI/NHEK.HiC-Benchmark.v3.tsv.gz
```

The first four are used for training, the last two are held out entirely for
testing. The file name before the first dot is read as the cell line name, so
keep the names as they are.

**2. Chromatin tracks.** Eight assays per cell line from ENCODE
(<https://www.encodeproject.org>): CTCF, DNase, H3K27ac, H3K27me3, H3K36me3,
H3K4me1, H3K4me3 and H3K9me3. Each is averaged into 500 base-pair bins and
saved as a PyTorch tensor file per chromosome, under
`data/genomic_data/processed/`. A JSON file maps each cell line and assay to its
tensor file:

```json
{
  "_location": "./data/genomic_data/processed",
  "HMEC": {
    "CTCF":   "narrowPeak_HMEC_CTCF.500bp.pt",
    "DNase":  "bigWig_HMEC_DNase.500bp.pt",
    "H3K27ac": "bigWig_HMEC_H3K27ac.500bp.pt"
  }
}
```

Save it as `data/genomic_data/CTCF_DNase_6histone_local.500.json`. A cell line
missing an assay is filled with zeros and a warning, so check the warnings on
the first run.

**3. Reference genome.** The hg19 build from UCSC:

```bash
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
gunzip hg19.fa.gz && mv hg19.fa data/hg19.fa
python -c "import pyfaidx; pyfaidx.Fasta('data/hg19.fa')"   # builds the index
```

If the genome or `pyfaidx` is missing, the sequence branch silently receives
placeholder sequences and the results are meaningless. The index step above
fails loudly if something is wrong, so run it.

## Running everything

```bash
python run.py
```

One command runs the whole plan: environment checks, the test suite, the split
audit, every training run, and every analysis. It takes about 25 hours of GPU
time at roughly 20 minutes per epoch.

**It is safe to interrupt.** Finished stages are recorded in
`results/run_state.json` and skipped on the next run. Inside a training run, the
complete state is written after every epoch, so an interrupted run resumes at
the epoch after the last completed one and loses at most 20 minutes. After a
power cut, just run `python run.py` again.

```bash
python run.py --check                # environment and data only, changes nothing
python run.py --list                 # what is done, pending or failed
python run.py --only leakage         # one stage
python run.py --from train_ka_seed0  # that stage and everything after it
python run.py --redo train_a_seed0   # force a stage to run again
python run.py --dry-run              # print the plan, run nothing
```

Each stage writes its console output to `results/logs/<stage>.log`. A failing
stage is reported and the run continues, so one failure does not cost a night.

## Running the stages by hand

The stages below are what `run.py` calls. Approximately 2 hours per five-epoch
run. The order matters: the first step costs nothing and can change how the rest
is interpreted.

### Step 1: audit the splits (processor only, about one minute)

```bash
python scripts/audit_leakage.py --config configs/slim_ka.yaml
```

Cross-cell-line evaluation holds out whole cell lines, but the held-out cell
lines still contribute every chromosome, and BENGI draws its enhancers from one
shared registry of candidate elements. The same genomic locus can therefore
carry a training pair in one cell line and a test pair in another. This script
measures how often that happens and writes an index marking the test pairs that
share no locus with training. Later steps score that subset separately, so the
question is answered with measurements rather than assumptions.

### Step 2: train each model on three seeds (about 24 hours)

```bash
for seed in 0 1 2; do
  python scripts/train.py --config configs/baseline.yaml   --seed $seed
  python scripts/train.py --config configs/slim_ka.yaml --seed $seed
  python scripts/train.py --config configs/slim_ga.yaml --seed $seed
done
python scripts/train.py --config configs/slim_a.yaml --seed 0
```

Each run writes to `results/<variant>/seed<N>/`. The seed fixes Python, NumPy
and PyTorch, the data loader workers, the shuffling order and the sample drawn
to fit the sequence encoder, so a run repeats on the same machine and library
versions. Add `--deterministic` to force deterministic kernels, which is slower
and unnecessary for the reported results.

Training resumes automatically. The full state, including the optimizer and the
scheduler, is written to `last.pt` after every epoch, to a temporary file that is
then moved into place so an interruption during the write cannot corrupt it. An
interrupted run continues at the epoch after the last completed one. Pass
`--no-resume` to ignore a saved state and start over.

### Step 3: how much does the chromatin branch carry? (about 4 hours)

```bash
python scripts/train.py --config configs/slim_ga.yaml --seed 0 \
    --modalities seq     --output-dir results/ga_seq_only
python scripts/train.py --config configs/slim_ga.yaml --seed 0 \
    --modalities seq+pos --output-dir results/ga_seq_pos
```

Testing on an unseen cell type while supplying that cell type's own chromatin
measurements is a weaker claim than sequence-only generalization. These two runs
quantify the difference. `seq` blanks the whole chromatin input; `seq+pos` keeps
only the channel marking where the enhancer and promoter sit. The architecture
is identical in both, so this is an input ablation and nothing else.

### Step 4: measure efficiency (about 10 minutes)

```bash
python scripts/benchmark_efficiency.py
```

Reports latency, throughput and peak video memory for every model at matched
batch size, using random tensors of the true shapes. Latency is timed with CUDA
events, because kernel launches are asynchronous and an ordinary timer measures
the queue rather than the work.

### Step 5: score everything (processor only, about two minutes)

```bash
python scripts/evaluate.py --run results/*/seed0 --leakage results/leakage \
    --out results/evaluation.json
python scripts/aggregate_seeds.py --reference baseline
python scripts/build_results.py
```

`evaluate.py` reads the saved predictions and reports metrics at the fixed 0.5
threshold, at a threshold chosen on validation, the calibration error, bootstrap
intervals, per cell line, and the leakage-disjoint subset. `aggregate_seeds.py`
reports per-seed values with means and standard deviations. `build_results.py`
writes `results/results.json` and regenerates the manuscript figures.

### Step 5b: is it overfitting? (processor only, seconds)

```bash
python scripts/check_overfitting.py --run results/*/seed0
```

Reports two gaps that are easy to confuse and mean different things.

The **overfitting gap** is the training score minus the validation score. Both
come from the same four cell lines, with validation on held-out chromosomes, so
this measures memorisation of training pairs. Under 0.10 in AUPR is small, 0.10
to 0.20 is normal at this budget, above 0.20 is large. The script also names the
epoch with the best validation score, and the epoch where validation loss first
rose while training loss was still falling.

The **transfer gap** is the validation score minus the test score. Validation and
test differ by cell line, not by chromosome, so this measures how much of what
the model learned is specific to the training cell types. This is the gap the
project is about, and it is usually the larger of the two. Training for fewer
epochs does not shrink it.

The distinction matters because the two call for opposite responses. In the runs
so far the baseline scores higher on validation (AUPR 0.620 against 0.606) and
lower on test (0.444 against 0.474), so the survival-gated encoder gives up a
little in-distribution accuracy and gains more out of it. A smaller transfer gap
at a similar validation score is the property this work claims to improve.

### Step 6: what does the memory select? (about 30 minutes)

```bash
python scripts/memory_writelog.py --run results/ga/seed0
```

Records which positions the survival gate writes, then tests whether those
positions carry more CTCF and open-chromatin signal than the positions dropped
from the same window. The comparison is paired within each window and the null
is built by reshuffling which bins count as written inside that same window, so
differences in overall activity between windows cannot produce an effect.

## Optional experiments

The configuration exposes the encoder's capacity directly, so these need no new
code:

```bash
# memory capacity
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --output-dir results/sweep_slots8   # then edit model.memory.bin_slots

# leave-one-chromosome-out, the strictest published split
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --split loco --loco-chrom chr1 --output-dir results/ka_loco_chr1
```

The gate's three scoring terms can each be switched off with
`model.memory.use_learned_score`, `use_novelty` and `use_prediction_error`, which
isolates their individual contributions.

## What each output file holds

| File                    | Contents                                                            |
| ----------------------- | ------------------------------------------------------------------- |
| `eval_results.npz`      | per-example validation and test predictions, labels, and the genomic coordinates of every test pair |
| `history.json`          | per-epoch training and validation metrics                            |
| `config_snapshot.yaml`  | the fully merged configuration, seed and split actually used         |
| `checkpoint.pt`         | the selected weights; not tracked, regenerated from the seed         |
| `encoder.pkl`           | the fitted sequence encoder; not tracked, regenerated from the seed  |

Predictions are saved with their coordinates so that every downstream analysis
reads a file rather than re-running the model. No analysis in this repository
needs a second training run.

## Reproducibility notes

- Every reported number is recomputed from `eval_results.npz`. Nothing is
  transcribed from a training log, and `build_results.py` reports a model as
  missing rather than filling it in from another source.
- Models are selected on validation AUROC plus AUPR, never on the test set.
  Decision thresholds are likewise chosen on validation. The manuscript reports
  both the fixed 0.5 threshold and the validation-chosen one, because the two
  differ substantially for some models.
- Repeated seeds measure variability between training runs. The bootstrap in
  `evaluate.py` measures variability from the finite test set. These are
  different quantities and both are reported.
- Runs on different GPU models or library versions will not match bit for bit,
  because mixed precision and autotuned kernels are not deterministic across
  hardware. The seed makes a run repeatable on one machine.

## Known limitations

- The chromatin branch reads the test cell type's own measured tracks. Step 3
  quantifies how much of the performance depends on that.
- The reported budget is five epochs. Longer training was not explored
  systematically.
- The benchmark is BENGI on hg19 with two held-out cell lines. Generalization to
  other benchmarks, assays and genome builds is untested.
- No comparison against selective state-space backbones or large pretrained
  genomic sequence models is included.
