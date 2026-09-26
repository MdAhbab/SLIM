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
does three things. Sliding-window attention mixes each token with its 64
nearest neighbours inside its own segment (enhancer DNA, promoter DNA, or
chromatin), computed block by block so its cost grows linearly with length. A
survival gate then scores every token and writes the eight highest-scoring
ones into a bounded memory of 16 slots: each survivor claims the free slot whose
content is most similar to its own, and a gated recurrent update writes it
there. Slots that are not written decay. Finally every token reads that memory
back by cross-attention. Distant information therefore travels through a small,
explicit memory rather than through all-to-all attention, and the selection
step is learned rather than fixed.

The survival score has three parts: a learned term, a novelty term measuring how
unlike the current memory a token is, and a prediction-error term measuring how
poorly the memory predicts that token, where each token is predicted from the
memory by a query built from its position alone and the predictor is trained by
its own loss. Selection is discrete in the forward pass and differentiable in
the backward pass through a straight-through estimator, written as a sum over
every token so that the gradient reaches the scores of tokens that were not
selected as well as those that were.

### Memory fixes, and the legacy comparison

The first KA results came from an encoder that did not do five of those things:
survivors always filled slots 0 to 7 in score order, so half the memory was
never written; local attention computed the full 628 by 628 score matrix and
then masked it; the prediction term compared one pooled vector per sequence
with every token, and nothing trained it; a layer norm over every slot undid the
decay; and the local window ran across the DNA and chromatin seams. Each is now
a switch under `model.memory` (`slot_addressing`, `prediction_target`,
`memory_norm`, `respect_segments`), documented in `src/memory_encoder.py`, and
the variant configurations turn them all on. `configs/slim_ka_legacy.yaml`
turns them off, which rebuilds the original model exactly, and `run.py` trains
it once so the effect of the fixes is measured rather than assumed. The
sliding-window attention needs no switch: it equals the masked version to
floating-point precision.

One problem is not fixed by default. The task gradient reaches the survival
score only at the tokens that were written, so the gate learns nothing about
the tokens it passes over. `gate_gradient: "all"` removes that restriction
without changing the forward pass, and `selection_noise` adds noisy top-k
selection, the remedy the original design proposed. Both are implemented and
tested, but in short training runs on the real data the first made the gate
write the same positions for every input sooner (pairwise overlap 0.93 and 0.83
on two seeds, against 0.33 and 0.45 without it) and the second did not help, so
the configurations keep the original rule. The write-log collapse check is
there to show whether the full-length runs stay input-driven.

### Pipeline fixes, and why the first runs overfit

In every run of the first full experiment, validation peaked after one or two
epochs while the training score kept climbing (AUROC 0.90 to 0.97). The
baseline did this exactly as much as the SLIM variants, so the cause was the
shared data and training pipeline, not the encoder. Five problems were found.
Each is now a switch whose original setting reproduces the old behaviour, and
`configs/base.yaml` turns every fix on for every model alike.

1. **The sequence encoding leaked training labels.** POCD-ND is fitted on
   labelled sequences and was fitted on training rows that were then trained
   on, so each of those rows was encoded with densities that counted its own
   label. At the real scale (5,000 plus 5,000 fitted sequences of 6,000 bp)
   a single linear statistic of the encoding separates the fitted rows
   perfectly even when the labels are random. `data.encoder_fit: crossfit`
   fits one encoder per chromosome half (odd and even numbers) and encodes
   every row with the encoder of the other half.
   `scripts/audit_encoder_leakage.py` measures the effect on the real data.
2. **The training assays did not match the test assay.** The loader reads
   every BENGI file whose name starts with a training cell line. With the
   ChIA-PET files present, as in the TransEPI copy of BENGI, 48 percent of the
   training rows and half of the validation rows came from ChIA-PET, whose
   positives are much closer together (median 31 kb for RNA polymerase II
   ChIA-PET against 90 to 230 kb for Hi-C). The test cell lines have Hi-C
   labels only. Distance alone scores validation AUROC 0.86 but test AUROC
   0.73, so validation rewarded a rule the test does not follow.
   `data.train_assays` and `training.valid_assays` now name the assays;
   validation always uses the Hi-C rows.
3. **Repeated pairs.** 38,787 training rows (12 percent) repeat a pair already
   present in another assay file of the same cell line, and 4,075 of those
   pairs carry conflicting labels. `data.dedup: union` merges them.
4. **Two augmentations were wrong.** Reverse-complementing the joined
   enhancer and promoter string moved the promoter into the enhancer's half,
   which neither the position-specific encoding nor the encoder's segments
   expect (`augmentation.rc_mode: segment` flips each sequence in place). The
   bin shift rolled the position channel along with the tracks, so it no
   longer matched the anchor indices (`augmentation.shift_mode: tracks`).
5. **No regularisation acted inside the budget.** The learning rate never
   changed, because the plateau rule needs five bad epochs; early stopping
   needed fifteen; weight decay was 1e-4. The recipe is now a warm-up and
   cosine schedule, weight decay 0.05 (not on biases or normalisation
   scales), stochastic depth 0.1 in both encoders, an exponential moving
   average of the weights for validation, selection and test, and early
   stopping after two epochs without improvement. `configs/slim_ka_old_recipe.yaml`
   keeps the corrected data with the old recipe, to separate the two.

The overfitting gap is also measured properly now: a fixed sample of 10,000
training rows is scored after every epoch exactly like validation
(`train_clean_*` in `history.json`), instead of the running score taken with
dropout and augmentation on.

Which training assays to use, and whether to keep the DNA branch, is decided
by four pilot runs of KA at seed 0 (Hi-C only or all assays, DNA on or off).
`scripts/choose_protocol.py` picks the one with the highest validation AUPR on
the Hi-C validation rows, without reading any test score, and every later run
applies that choice with `--protocol results/protocol.json`. The pilots that
lose become ablations.

`scripts/audit_dataset.py` also reports what a model with no network at all
reaches: ranking by distance gives test AUROC 0.73 and AUPR 0.20, and a
logistic regression on distance plus how often the gene and the enhancer
interacted in the training cell lines, fitted on Hi-C rows, gives 0.81 and
0.41. Every trained model should be read against those two numbers.

## Models

Four models share the branches, the pooling, the prediction heads, the data
pipeline, the optimizer and the training budget. The encoder is the only thing
that changes, and among the three SLIM variants only the feed-forward
sublayer inside the encoder changes.

| Name      | Encoder                        | Feed-forward sublayer            | Parameters |
| --------- | ------------------------------ | -------------------------------- | ---------: |
| baseline  | global self-attention          | spline (Kolmogorov-Arnold) layer |  3,529,796 |
| A         | survival-gated memory encoder  | rectified layer, width 720       |  4,392,095 |
| KA        | survival-gated memory encoder  | spline layer, width 64           |  4,302,995 |
| GA        | survival-gated memory encoder  | gated linear layer, width 480    |  4,392,815 |
| KA legacy | original memory rules, one run | spline layer, width 64           |  4,251,155 |

The memory fixes add one 180 by 96 position query per layer, 51,840 parameters
in all, to every SLIM variant alike.

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
  bengi.py             the one BENGI parser, repeat merging, assay selection
  epi_data_pipeline.py reads BENGI pairs and binned chromatin tracks
  dataset.py           PyTorch dataset, augmentation, sequence encoding
  encoding.py          the position-aware trinucleotide encoder, cross-fitted
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
  audit_dataset.py     repeats, anchor reuse, distance and locus baselines
  audit_encoder_leakage.py  label leakage through the sequence encoding
  choose_protocol.py   picks training assays and inputs from the pilots
  audit_leakage.py     genomic overlap between the splits
  check_overfitting.py the overfitting gap and the transfer gap
  aggregate_seeds.py   per-seed values, means, paired differences
  benchmark_efficiency.py  latency, throughput, peak memory
  memory_writelog.py   what the survival gate selects
  build_results.py     tables and figures from saved predictions
  make_tables.py       the manuscript tables as LaTeX fragments
results/          one directory per model, holding per-example predictions
tests/            run these before any long job
```

## Hardware and installation

Developed and run on a single NVIDIA RTX 5070 Ti with 16 GB of video memory and
32 GB of system memory. The batch size is 32. At 64 the KA variant reserves
about 18.7 GB, spills into shared system memory, and runs roughly 2.6 times
slower per sample; at 32 the heaviest variant reserves under 10 GB. This card
needs a PyTorch build for CUDA 12.8 or later, such as the cu130 wheels; older
builds install but carry no kernels for it.

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
python -m pytest -q                # 53 tests, about 20 seconds
python -m pytest -q -m slow        # end to end on synthetic data, about 45 seconds
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
testing. The file name before the first dot is read as the cell line name and
the part after it as the assay, so keep the names as they are. Every file of a
training cell line is read, so if the ChIA-PET benchmarks are present too
(`GM12878.CTCF-ChIAPET-Benchmark.v3.tsv.gz` and the like, as in the TransEPI
copy of BENGI), `data.train_assays` decides whether they are used; the pilot
runs choose.

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

One command runs the whole plan: environment checks, the test suite, the two
data audits, the four protocol pilots and the choice between them, every
training run, and every analysis. It takes about 34 hours of GPU time; the
estimate per stage is shown by `python run.py --list`.

Results from an earlier plan must be moved out of the way first, or their
stages would be skipped as already finished:

```bash
python run.py --archive before_pipeline_fixes
```

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
python run.py --archive NAME         # move all results to results/archive/NAME/
```

Each stage writes its console output to `results/logs/<stage>.log`. A failing
stage is reported and the run continues, so one failure does not cost a night.

## Running the stages by hand

The stages below are what `run.py` calls. Approximately 2 hours per five-epoch
run. The order matters: the first step costs nothing and can change how the rest
is interpreted.

### Step 0: audit the data (processor only, about twenty minutes)

```bash
python scripts/audit_dataset.py
python scripts/audit_encoder_leakage.py
```

The first needs only the BENGI files and reports repeated pairs, how often
anchors recur, and the distance-only and locus-prior baselines. The second
needs the genome and measures how strongly each way of fitting the sequence
encoder carries training labels into the encoding.

### Step 0b: choose the protocol (about seven GPU hours)

```bash
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --train-assays HiC --modalities all --output-dir results/pilot_hic_dna
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --train-assays HiC --modalities epi --output-dir results/pilot_hic_nodna
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --train-assays all --modalities all --output-dir results/pilot_all_dna
python scripts/train.py --config configs/slim_ka.yaml --seed 0 \
    --train-assays all --modalities epi --output-dir results/pilot_all_nodna
python scripts/choose_protocol.py --pilots results/pilot_* --out results/protocol.json
```

All four are validated on the same Hi-C rows. The choice uses validation AUPR
only. Add `--keep-dna` to `choose_protocol.py` to consider only the pilots
with the DNA branch on.

### Step 1: audit the splits (processor only, about one minute)

```bash
python scripts/audit_leakage.py --config configs/slim_ka.yaml \
    --protocol results/protocol.json
```

Cross-cell-line evaluation holds out whole cell lines, but the held-out cell
lines still contribute every chromosome, and BENGI draws its enhancers from one
shared registry of candidate elements. The same genomic locus can therefore
carry a training pair in one cell line and a test pair in another. This script
measures how often that happens and writes an index marking the test pairs that
share no locus with training. Later steps score that subset separately, so the
question is answered with measurements rather than assumptions.

### Step 2: train each model (about 18 hours)

```bash
P="--protocol results/protocol.json"
for seed in 0 1 2; do
  for config in baseline slim_ka slim_ga slim_a; do
    python scripts/train.py --config configs/$config.yaml --seed $seed $P
  done
done
for seed in 3 4; do
  python scripts/train.py --config configs/baseline.yaml --seed $seed $P
  python scripts/train.py --config configs/slim_ka.yaml  --seed $seed $P
done
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

### Step 3: ablations of variant KA (about 8 hours)

```bash
P="--protocol results/protocol.json"
python scripts/train.py --config configs/slim_ka_old_recipe.yaml    --seed 0 $P --output-dir results/ka_old_recipe/seed0
python scripts/train.py --config configs/slim_ka_legacy.yaml        --seed 0 $P --output-dir results/ka_legacy/seed0
python scripts/train.py --config configs/slim_ka.yaml --seed 0 $P --modalities seq --output-dir results/ka_dna_geometry/seed0
python scripts/train.py --config configs/slim_ka_random_select.yaml --seed 0 $P --output-dir results/ka_random_select/seed0
python scripts/train.py --config configs/slim_ka_no_readback.yaml   --seed 0 $P --output-dir results/ka_no_readback/seed0
python scripts/train.py --config configs/slim_ka_learned_only.yaml  --seed 0 $P --output-dir results/ka_learned_only/seed0
```

Each changes one thing: the training recipe, the memory rules, the chromatin
tracks, how survivors are chosen, whether the memory is read back, or which
terms the survival score uses. Together with the pilots, which vary the
training assays and the DNA branch, they give every ablation in the
manuscript.

`--modalities seq` blanks the whole chromatin input, but it is not
sequence-only: the head still reads the tokens at the enhancer and promoter
positions, and the window is centred on the pair, so the pair's distance
remains visible. It is reported as "DNA and pair geometry". The distance-only
baseline from Step 0 is the floor it has to beat. `--modalities epi` does the
opposite and switches the DNA branch off.

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

The distinction matters because the two call for opposite responses. When a run
records `train_clean_*`, the training score is that clean sample, scored like
validation; older runs only have the running score, and the report says which
it used. Since validation now uses Hi-C rows only, whose positive rate (about
4 percent) is lower than the test cell lines' (about 11 percent), absolute
validation and test AUPR are not comparable; compare transfer gaps between
models, not across the two splits.

### Step 6: what does the memory select? (about 30 minutes)

```bash
python scripts/memory_writelog.py --run results/ka/seed0
```

Records which positions the survival gate writes, then tests whether those
positions carry more CTCF and open-chromatin signal than the positions dropped
from the same window. The comparison is paired within each window and the null
is built by reshuffling which bins count as written inside that same window, so
differences in overall activity between windows cannot produce an effect.

It also checks for collapse, which enrichment alone cannot detect: how often
each position is chosen across inputs, how much two inputs' written sets
overlap compared with random sets of the same size, and how many memory slots
are ever written. A mean overlap of one half or more is flagged. Run it on
`results/ka_legacy/seed0` as well. The original gate is strongly positional:
on the first trained KA model six positions were written by at least 90
percent of test windows, and two windows' written sets overlapped 22 times
more than random sets of the same size would.

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
isolates their individual contributions. The memory switches described above
can likewise be turned off one at a time to attribute the legacy comparison to
a single fix.

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
- The budget is at most eight epochs with early stopping. Longer training was
  not explored systematically.
- Holding out cell lines does not hold out loci: a regression on how often a
  gene and an enhancer interacted in the training cell lines already reaches
  test AUPR 0.41 with distance. Step 1 and the leakage-disjoint subset measure
  how much of a model's score depends on shared loci.
- The benchmark is BENGI on hg19 with two held-out cell lines. Generalization to
  other benchmarks, assays and genome builds is untested.
- No comparison against selective state-space backbones or large pretrained
  genomic sequence models is included.
