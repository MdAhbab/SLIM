# SLIM results

Cross-cell-line enhancer-promoter interaction prediction. Every number here comes from the files this run wrote in `results/`; the last section says which file holds which.

Run on one NVIDIA GeForce RTX 5070 Ti, PyTorch 2.14.0+cu130, finished 23 September 2026: twenty-one stages in 15 hours 53 minutes, thirteen trained models. Two of those runs, the baseline at seed 0 and KA legacy, were trained in an earlier session under identical settings and reused rather than repeated.

## What was run

**Protocol.** Four cell lines are used for training (GM12878, HeLa, K562, IMR90). Two chromosomes inside them (chr11, chr17) are held out for validation, which is what selects the checkpoint and the decision threshold. The test set is two cell lines the model never sees, HMEC and NHEK: 37,707 candidate pairs, 4,106 of them interacting (10.9 percent). Nothing about the test cell lines enters training or model selection.

**Budget, shared by every model.** Five epochs, batch size 32, AdamW at learning rate 1e-4 with weight decay 1e-4, mixed precision, and the checkpoint chosen by validation AUROC plus AUPR. Because the budget, the data pipeline and everything outside the compared component are shared, a difference between two models is attributable to that component.

**Models.**

| model | encoder | feed-forward | parameters |
| --- | --- | --- | ---: |
| baseline | global self-attention | spline (KAN) | 3,529,796 |
| A | survival-gated memory | rectified, width 720 | 4,392,095 |
| KA | survival-gated memory | spline, width 64 | 4,302,995 |
| GA | survival-gated memory | gated linear, width 480 | 4,392,815 |
| KA legacy | survival-gated memory, original rules | spline, width 64 | 4,251,155 |

KA legacy rebuilds the encoder as it was before the memory fixes described later, so the two KA rows isolate those fixes.

## Headline results

Test set (HMEC and NHEK together), at the fixed 0.5 threshold. Mean over seeds, with the standard deviation across seeds where there is more than one.

| model | seeds | AUROC | AUPR | balanced accuracy | F1 | MCC |
| --- | ---: | --- | --- | --- | --- | --- |
| baseline | 3 | 0.8448 ± 0.0128 | 0.4872 ± 0.0354 | 0.6495 ± 0.0186 | 0.4188 ± 0.0379 | 0.3877 ± 0.0375 |
| KA | 3 | 0.8538 ± 0.0058 | 0.5090 ± 0.0215 | 0.7069 ± 0.0288 | 0.4816 ± 0.0315 | 0.4234 ± 0.0307 |
| GA | 3 | 0.8501 ± 0.0192 | 0.4832 ± 0.0446 | 0.6878 ± 0.0238 | 0.4579 ± 0.0306 | 0.4033 ± 0.0455 |
| A | 1 | 0.8421 | 0.4576 | 0.6380 | 0.3894 | 0.3486 |
| KA legacy | 1 | 0.8469 | 0.4957 | 0.6552 | 0.4374 | 0.4154 |

KA is the best of the three variants on AUROC, AUPR and MCC, and the most consistent across seeds (AUROC spread 0.006 against 0.013 for the baseline and 0.019 for GA). **The spread is the thing to keep in view**: for AUPR it is larger than the gap between models, so these means describe what was observed and do not establish that KA is better.

### Per seed

| model | seed | AUROC | AUPR | balanced accuracy | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0 | 0.8596 | 0.5270 | 0.6700 | 0.4614 | 0.4309 |
| baseline | 1 | 0.8370 | 0.4594 | 0.6336 | 0.3889 | 0.3636 |
| baseline | 2 | 0.8378 | 0.4751 | 0.6450 | 0.4060 | 0.3686 |
| KA | 0 | 0.8519 | 0.5223 | 0.7306 | 0.4871 | 0.4199 |
| KA | 1 | 0.8602 | 0.5204 | 0.7151 | 0.5100 | 0.4557 |
| KA | 2 | 0.8491 | 0.4841 | 0.6749 | 0.4478 | 0.3947 |
| GA | 0 | 0.8325 | 0.4466 | 0.6694 | 0.4230 | 0.3584 |
| GA | 1 | 0.8706 | 0.5329 | 0.6794 | 0.4799 | 0.4493 |
| GA | 2 | 0.8472 | 0.4701 | 0.7148 | 0.4709 | 0.4022 |
| A | 0 | 0.8421 | 0.4576 | 0.6380 | 0.3894 | 0.3486 |
| KA legacy | 0 | 0.8469 | 0.4957 | 0.6552 | 0.4374 | 0.4154 |

Seed choice moves AUPR by up to 0.09 within one model (GA: 0.4466 to 0.5329). Any comparison smaller than that needs more seeds before it can be called a difference.

### Difference from the baseline, on the seeds both models ran

| model | AUROC | AUPR | balanced accuracy | F1 | MCC | seeds |
| --- | --- | --- | --- | --- | --- | ---: |
| KA | +0.0090 ± 0.0156 | +0.0218 ± 0.0346 | +0.0574 ± 0.0260 | +0.0629 ± 0.0511 | +0.0357 ± 0.0522 | 3 |
| GA | +0.0053 ± 0.0305 | -0.0040 ± 0.0769 | +0.0383 ± 0.0358 | +0.0392 ± 0.0684 | +0.0156 ± 0.0806 | 3 |
| A | -0.0174 | -0.0694 | -0.0320 | -0.0720 | -0.0822 | 1 |
| KA legacy | -0.0127 | -0.0313 | -0.0147 | -0.0240 | -0.0154 | 1 |

KA improves on the baseline on every metric **on average**. Balanced accuracy (+0.057) and F1 (+0.063) improve in every seed; AUROC, AUPR and MCC improve on average but not in every seed. GA is mixed, and variant A is worse on the one seed it ran.

Read the single-seed rows carefully. Seed 0 is the baseline's strongest seed, and on that seed both KA variants trail it on AUROC and AUPR. Legacy KA is below the baseline there and below fixed KA on every metric, so what seed 0 shows is that the memory fixes improve KA, not that either KA beats the baseline.

### By cell line

Mean over seeds, threshold 0.5.

| model | HMEC AUROC | HMEC AUPR | NHEK AUROC | NHEK AUPR |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.8839 | 0.5372 | 0.8034 | 0.4284 |
| KA | 0.8844 | 0.5420 | 0.8220 | 0.4712 |
| GA | 0.8869 | 0.5232 | 0.8086 | 0.4370 |
| A | 0.8766 | 0.5148 | 0.8076 | 0.3866 |
| KA legacy | 0.8808 | 0.5394 | 0.8081 | 0.4486 |

NHEK is the harder cell line for every model, by 0.06 to 0.08 AUROC and 0.07 to 0.13 AUPR. That ordering is a property of the data, not of any model. KA has the smallest HMEC-to-NHEK drop of the three variants, which is the same consistency its seed spread shows.

## Threshold choice and calibration

The 0.5 threshold is arbitrary for a task with 10.9 percent positives. Each run therefore also reports the threshold that maximises MCC on validation, which is an operating point a user could actually pick without seeing the test set.

| run | threshold from validation | MCC at 0.5 | MCC at that threshold | ECE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline seed 0 | 0.117 | 0.4309 | 0.4361 | 0.0535 | 0.0755 |
| baseline seed 1 | 0.146 | 0.3636 | 0.3764 | 0.0571 | 0.0810 |
| baseline seed 2 | 0.150 | 0.3686 | 0.4002 | 0.0600 | 0.0820 |
| KA seed 0 | 0.411 | 0.4199 | 0.4145 | 0.0715 | 0.0879 |
| KA seed 1 | 0.145 | 0.4557 | 0.4067 | 0.0553 | 0.0800 |
| KA seed 2 | 0.361 | 0.3947 | 0.3830 | 0.0305 | 0.0781 |
| GA seed 0 | 0.349 | 0.3584 | 0.3539 | 0.0544 | 0.0859 |
| GA seed 1 | 0.163 | 0.4493 | 0.4556 | 0.0429 | 0.0731 |
| GA seed 2 | 0.534 | 0.4022 | 0.4006 | 0.0858 | 0.1015 |
| A seed 0 | 0.248 | 0.3486 | 0.3690 | 0.0389 | 0.0797 |
| KA legacy seed 0 | 0.249 | 0.4154 | 0.3925 | 0.0360 | 0.0753 |

Two things worth stating plainly. The validation-chosen threshold lands anywhere from 0.117 to 0.534 and does **not** reliably beat 0.5 on the test set: it improves MCC for the baseline in all three seeds but lowers it for KA in all three. Reporting both, as the tables do, is the honest course, and the fixed 0.5 threshold is not flattering anyone.

Calibration is the one place the memory fixes look worse. Fixed KA ranges from 0.031 to 0.072 ECE across seeds against legacy KA's 0.036, so it is worse on two of the three seeds and better on one. The variability itself is the finding; it should be reported rather than averaged away.

## Does the result depend on loci seen in training?

Test pairs are split by whether their genomic locus also appears in the training data. If a model were scoring well by memorising territory, it would do markedly better on the reused half.

| run | AUROC, unseen loci | AUROC, reused loci | difference |
| --- | ---: | ---: | ---: |
| baseline seed 0 | 0.8532 | 0.8565 | -0.0033 |
| baseline seed 1 | 0.8295 | 0.8367 | -0.0073 |
| baseline seed 2 | 0.8358 | 0.8332 | +0.0026 |
| KA seed 0 | 0.8462 | 0.8485 | -0.0023 |
| KA seed 1 | 0.8462 | 0.8598 | -0.0136 |
| KA seed 2 | 0.8415 | 0.8476 | -0.0061 |
| GA seed 0 | 0.8201 | 0.8320 | -0.0119 |
| GA seed 1 | 0.8518 | 0.8718 | -0.0200 |
| GA seed 2 | 0.8387 | 0.8457 | -0.0070 |
| A seed 0 | 0.8354 | 0.8387 | -0.0033 |
| KA legacy seed 0 | 0.8374 | 0.8446 | -0.0072 |

The differences run from -0.020 to +0.003, which is noise. Performance on territory the model has never seen matches performance on territory it has. Memorised loci are not carrying the result.

**A caveat about the AUPR version of this comparison.** `evaluation.json` also reports "AUPR on reused loci minus disjoint loci", which looks alarming (around -0.12). That number is confounded: the two subsets have different positive rates (17.1 percent unseen against 9.2 percent reused), and AUPR rises with prevalence on its own. AUROC does not have that problem, so the table above is the comparison to quote.

## Overfitting and transfer

Two different gaps matter, and they call for opposite responses. The **overfitting gap** is training minus validation AUPR inside the training cell lines. The **transfer gap** is validation minus test AUPR, which is what it costs to move to an unseen cell type.

| run | best epoch | overfitting gap there | gap at epoch 5 | transfer gap |
| --- | ---: | ---: | ---: | ---: |
| baseline seed 0 | 3 of 5 | +0.1828 | +0.3555 | +0.0769 |
| baseline seed 1 | 3 of 5 | +0.1691 | +0.2532 | +0.1569 |
| baseline seed 2 | 4 of 5 | +0.2023 | +0.3093 | +0.1452 |
| KA seed 0 | 2 of 5 | +0.1421 | +0.3257 | +0.0756 |
| KA seed 1 | 4 of 5 | +0.2316 | +0.2746 | +0.0938 |
| KA seed 2 | 1 of 5 | +0.0163 | +0.2952 | +0.1414 |
| GA seed 0 | 2 of 5 | +0.1323 | +0.2963 | +0.1689 |
| GA seed 1 | 4 of 5 | +0.2520 | +0.2819 | +0.0714 |
| GA seed 2 | 4 of 5 | +0.2470 | +0.3032 | +0.1383 |
| A seed 0 | 2 of 5 | +0.1200 | +0.3008 | +0.1625 |
| KA legacy seed 0 | 2 of 5 | +0.1242 | +0.3082 | +0.1182 |

Every model overfits the training cell lines, the baseline as much as the others, and by epoch 5 the gap reaches 0.25 to 0.36 everywhere. Three points follow:

1. **It is not in the reported numbers.** Validation peaks at epoch 1 to 4 and the saved checkpoint is that epoch, so the later, more overfit epochs are visible in the logs but score nothing.
2. **It cannot inflate the test score**, which is measured on cell lines absent from both training and model selection. The leakage split above is the direct check, and it comes back clean.
3. **It is uniform across models**, so the comparisons are unaffected.

What it does cost: three of five epochs are wasted in most runs, and the choice of peak epoch adds seed-to-seed noise. KA seed 2 peaked at epoch 1, making it effectively a one-epoch model and KA's weakest run. Note also that `patience: 15` with a five-epoch budget means early stopping can never fire; checkpoint selection is doing that work instead.

## Ablations

### What the inputs contribute

GA, seed 0, with parts of the input removed.

| input | AUROC | AUPR |
| --- | ---: | ---: |
| everything | 0.8325 | 0.4466 |
| sequence and position, no chromatin tracks | 0.7532 | 0.2322 |
| DNA sequence only | 0.7212 | 0.2038 |

The chromatin tracks carry most of the signal: removing them halves AUPR (0.447 to 0.232), and DNA sequence alone reaches 0.204. Whatever the encoder is doing, it is doing it mostly with the chromatin branch. This also sets a floor: any claim about sequence modelling has to beat 0.204.

### What the feed-forward sublayer contributes

A, KA and GA differ only in that sublayer. Over the seeds available, KA (spline) leads, GA (gated linear) follows, and A (rectified, one seed) is last. A and GA hold the same number of parameters by construction, so that particular comparison is about the layer type rather than capacity.

### What the memory fixes changed

KA and KA legacy differ only in the memory rules. On seed 0:

| | KA legacy | KA fixed |
| --- | ---: | ---: |
| test AUROC | 0.8469 | 0.8519 |
| test AUPR | 0.4957 | 0.5223 |
| test MCC | 0.4154 | 0.4199 |
| calibration error | 0.0360 | 0.0715 |
| memory slots ever written | 8 of 16 | 16 of 16 |
| positions ever written | 184 of 628 | 309 of 628 |
| positions written for at least 90% of inputs | 6 | 0 |
| overlap between two inputs' writes | 0.411 | 0.196 |

The accuracy difference is modest and rests on one seed. The behavioural difference is not: the fixed encoder uses its whole memory, writes a wider and more input-dependent set of positions, and no longer has positions that win for nearly every input.

## What the memory selects

One inference pass over the test cell lines, recording which positions each layer writes into memory (1,280 windows).

| | KA legacy | KA fixed | GA fixed |
| --- | ---: | ---: | ---: |
| writes landing in the chromatin branch | 96.1% | 89.7% | 82.0% |
| mean distance from a write to the nearer anchor | 90 tokens | 44 tokens | 78 tokens |
| memory slots ever written | 8 of 16 | 16 of 16 | 16 of 16 |
| positions ever written | 184 of 628 | 309 of 628 | 266 of 628 |
| overlap between two inputs' writes | 0.411 | 0.196 | 0.317 |

Writes concentrate near the enhancer and promoter anchors: the fixed KA averages 44 tokens (about 222 kb) from the nearer anchor, against 125 tokens for writes scattered at random. Two inputs still share far more written positions than chance (0.018 for random sets), which is expected for a task with strong positional structure, but the fixed models are much less positional than the legacy one.

### Do the selected positions carry regulatory signal?

Paired inside each window: the chromatin signal in written bins against unwritten bins of the same window, with a permutation test that reshuffles which bins count as written inside that window.

| track | written | unwritten | ratio | p |
| --- | ---: | ---: | ---: | ---: |
| CTCF | 0.0637 | 0.0536 | 1.188 | 0.0005 |
| DNase | 0.0889 | 0.0706 | 1.259 | 0.0005 |
| H3K27ac | 0.0856 | 0.0592 | 1.447 | 0.0005 |
| H3K27me3 | 0.1099 | 0.1260 | 0.872 | 0.0005 |
| H3K4me3 | 0.0858 | 0.0566 | 1.515 | 0.0005 |

The gate prefers bins carrying active marks and avoids the repressive one. The strongest enrichments are H3K4me3 (active promoters, 1.5 times) and H3K27ac (enhancer activity, 1.4 times), followed by DNase (open chromatin) and CTCF (architectural binding); H3K27me3 (repression) is depleted at 0.87. That is the direction regulatory biology predicts, and it is evidence the selection is not arbitrary. Quote the ratios rather than the p-values: with 1,280 paired windows even a small effect returns the smallest p the permutation test can produce.

## Cost

Measured on the NVIDIA GeForce RTX 5070 Ti at batch size 32, median of 30 repeats.

| model | parameters | inference | pairs/s | training step | peak training memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 3,529,796 | 54.3 ms | 589 | 149.4 ms | 6.4 GB |
| A | 4,392,095 | 23.2 ms | 1382 | 65.5 ms | 3.1 GB |
| KA | 4,302,995 | 56.9 ms | 563 | 151.3 ms | 6.7 GB |
| GA | 4,392,815 | 23.3 ms | 1372 | 66.7 ms | 3.1 GB |

A and GA are about 2.3 times faster than the baseline and use half its training memory. KA is not: its spline layer costs roughly what global attention saves at this sequence length, so KA buys its accuracy with compute rather than saving any. **The encoder does not make the model faster at 628 tokens**; the sliding-window attention is linear in length, so its advantage would appear at longer inputs, which this run does not test.

## How far these numbers go

- **Three seeds cannot establish significance.** For AUPR the seed spread exceeds the gap between models. Balanced accuracy and F1 improve for KA in every seed, which is the strongest statement the data supports.
- **Variant A, KA legacy and both input ablations ran on one seed each.** Their differences are indicative only.
- **The write-log analysis covers three trained models**, all on the test cell lines.
- **Calibration varies by seed** and is worse for fixed KA than legacy KA on two seeds of three.
- **Efficiency was measured at one sequence length**, so the linear-cost claim for the encoder is architectural, not demonstrated here.
- **Not run:** capacity sweeps over memory slots and window size, the leave-one-chromosome-out split, comparisons against other long-context models or pretrained models, and motif or loop-anchor analysis of the selected positions.

## Where each number comes from

| file | holds |
| --- | --- |
| `results/seed_summary.json` | per-seed values, means, spreads, paired differences |
| `results/evaluation.json` | thresholds, calibration, bootstrap intervals, per cell line, leakage split |
| `results/overfitting.json` | per-epoch curves, overfitting and transfer gaps |
| `results/efficiency.json` | latency, throughput, peak memory |
| `results/<model>/seed<N>/eval_results.npz` | every per-example prediction, with genomic coordinates |
| `results/<model>/seed<N>/history.json` | per-epoch training and validation metrics |
| `results/<model>/seed<N>/config_snapshot.yaml` | the exact settings that produced the run |
| `results/{ka,ka_legacy,ga}/seed0/writelog.json` | memory write log and collapse check |
| `results/archive/before_memory_fixes/` | the earlier batch-64 runs, kept for reference |
| `results/logs/` | full console output of every stage |
| `figures/` | the three figures, as PDF and PNG |

