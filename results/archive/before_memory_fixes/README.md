# Results from before the memory-encoder fixes

Kept for reference; nothing in the pipeline reads this folder.

`baseline/` and `ka/` hold the runs whose numbers the manuscript first
reported, with `evaluation.json` and `results.json` computed from them. They
were trained at batch size 64 by the earlier code, and the KA model is the
original memory encoder: survivors written to slots 0-7 in score order, a
pooled prediction term, a layer norm that undoes the decay, and local
attention that crosses the DNA and chromatin seams. `configs/slim_ka_legacy.yaml`
rebuilds that encoder.

They were moved here because `run.py` discovers runs by globbing
`results/*/eval_results.npz`, which would have mixed these batch-64 runs into
the evaluation of the new ones.

`ga_seed0_partial/` is a GA run that stopped after three of five epochs while
it was still using the original memory rules. It cannot be resumed into the
fixed model and has no test predictions.
