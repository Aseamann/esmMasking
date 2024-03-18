# esmMasking
Script to perform zero-shot prediction of mutational effects with ESM2 pLM


## References
Implementation of: J. Meier doi: 10.1101/2021.07.09.450648

Modified script: https://huggingface.co/blog/AmelieSchreiber/mutation-scoring


## Example Usage:
python esm_masking.py -i rcsb_pdb_7XE4.fasta -m 639,641,643


### Highlight known mutations...
---
#### Uncomment line 123 -- Update dictionary with mutated residues
---
Packages utilized:
torch - 2.1.1+cu118; transformers - 4.35.2; matplotlib - 3.8.3; numpy - 1.24.1; seaborn - 0.13.2
