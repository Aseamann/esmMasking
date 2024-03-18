###################################################################
#                          esm_masking.py                         #
#                          --------------                         #
# Script to perform zero-shot prediction of mutational effects    #
# with ESM2 pLM.                                                  #
# Written by: Austin Seamann, 2024                                #
# Implementation of: J. Meier doi: 10.1101/2021.07.09.450648      #
# Modified script:                                                #
# https://huggingface.co/blog/AmelieSchreiber/mutation-scoring    #
###################################################################

from transformers import AutoTokenizer, EsmForMaskedLM
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle as pkl
import os
import argparse

VERBOSE = False

def generate_heatmap(protein_sequence, mutation_positions, known_mutations={}):
    global VERBOSE
    # Load the model and tokenizer
    model_name = "facebook/esm2_t36_3B_UR50D"  # Find other models online
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name)

    # Tokenize the input sequence
    input_ids = tokenizer.encode(protein_sequence, return_tensors="pt")
    sequence_length = input_ids.shape[1] - 2  # Excluding the special tokens

    # List of amino acids
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    # Read in heatmap pickle if found
    if os.path.exists("heatmap_data.pkl"):
        with open("heatmap_data.pkl", "rb") as f:
            heatmap = pkl.load(f)
    else:  # Run ESM and collect data
        # Initialize heatmap
        heatmap = np.zeros((20, len(mutation_positions)))

        # Calculate LLRs for each position and amino acid
        for i in range(len(mutation_positions)):
            # Mask the target position
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, mutation_positions[i]] = tokenizer.mask_token_id
            
            # Get logits for the masked token
            with torch.no_grad():
                logits = model(masked_input_ids).logits
                
            # Calculate log probabilities
            probabilities = torch.nn.functional.softmax(logits[0, mutation_positions[i]], dim=0)
            log_probabilities = torch.log(probabilities)
            
            # Get the log probability of the wild-type residue
            wt_residue = input_ids[0, mutation_positions[i]].item()
            log_prob_wt = log_probabilities[wt_residue].item()
            
            # Calculate LLR for each variant
            for j, amino_acid in enumerate(amino_acids):
                log_prob_mt = log_probabilities[tokenizer.convert_tokens_to_ids(amino_acid)].item()
                heatmap[j, i] = log_prob_mt - log_prob_wt
    
        # Save heatmap to pickle
        with open("heatmap_data.pkl", "wb") as f:
            pkl.dump(heatmap, f)

    # Visualize the heatmap
    if VERBOSE:
        print("Making heatmap")
    plt.figure(figsize=(15, 5), dpi=300)
    # playful color map
    sns.heatmap(heatmap, xticklabels=list(mutation_positions), yticklabels=amino_acids, cmap="inferno")
    # Indicate the wild-type residue on heatmap (0.5 offset to center the marker on the cell)
    plt.plot([mutation_positions.index(i) + 0.5 for i in mutation_positions], [amino_acids.index(protein_sequence[i-1]) + 0.5 for i in mutation_positions], 'wo')
    # Indicate the known mutations on heatmap - using red x
    if known_mutations != {}:
        for pos, mut in known_mutations.items():
            plt.plot(mutation_positions.index(pos) + 0.5, amino_acids.index(mut) + 0.5, 'bx')
    # Presentation size font and labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Position in Protein Sequence", fontsize=16)
    plt.ylabel("Amino Acid Mutations", fontsize=16)
    plt.title("Predicted Effects of Mutations on Protein Sequence (LLR)", fontsize=20)
    # Add legend indicating the wild-type residue as wo and known mutations as bx - place it outside the plot
    plt.legend(["Wild-type Residue", "Known Mutation"])
    # plt.colorbar(label="Log Likelihood Ratio (LLR)")
    plt.savefig("/projects/f_sdk94_1/als515/AIRevolution/heatmap.png")

def parse_args():
    parser = argparse.ArgumentParser("Generate heatmap of predicted effects of mutations on protein sequence using ESM2 pLM.")
    parser.add_argument("-i", "--input", help="Input fasta file or protein sequence", type=str, required=True)
    parser.add_argument("-m", "--mutations", help="Comma separated list of mutation positions", type=str, required=True)
    parser.add_argument("-v", "--verbose", help="Print verbose output", action="store_true", default=False)
    parser.add_argument("-o", "--output", help="Output file/path of heatmap", type=str, default="heatmap.png")
    args = parser.parse_args()
    return args

def main(args):
    # Example usage: python esm_masking.py -i rcsb_pdb_7XE4.fasta -m 639,641,643
    global VERBOSE
    VERBOSE = args.verbose
    # Prepare data
    # Read in sequence from fasta file
    if os.path.exists(args.input):
        with open('rcsb_pdb_7XE4.fasta', 'r') as file:
            seq_in = "".join(file.readlines()[1:]).strip('\n')
    else:
        seq_in = args.input.strip('\n').strip(' ')
    if VERBOSE:
        print("seq_in:", seq_in)
    mutation_postions = args.mutations.split(',')
    if VERBOSE:
        print("mutation_postions:", mutation_postions)
    # mutation_postions = [639, 641, 643, 646, 647, 692, 694, 695, 696,
    #                      697, 698, 699, 1354, 1355, 1357, 1369]
    # Known mutations: (position: mutation)
    # k_mutations = {639: "S", 643: "P"} - provide as third argument
    generate_heatmap(seq_in, mutation_postions)

if __name__ == "__main__":
    args = parse_args()
    main(args)
