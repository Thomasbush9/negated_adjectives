import marimo

__generated_with = "0.13.6"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import os
    from pathlib import Path
    from tqdm import tqdm
    import dspy
    import torch 
    from torch.utils.data import DataLoader, Dataset
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd
    from mofresh import anywidget, refresh_matplotlib
    from dotenv import load_dotenv
    import einops
    return Path, dspy, load_dotenv, mo, os, pd, tqdm


@app.cell
def _():
    from datetime import datetime
    from dspy import Predict
    return


@app.cell
def _(mo):
    mo.md(r"""## Dataset generation:""")
    return


@app.cell
def _(Path, load_dotenv, os):
    #load env vars
    load_dotenv(dotenv_path=".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_PERSONAL")
    triplets_csv_path = Path(os.getenv("TOKENS"))
    main_dir = Path(os.getenv("MAIN_DIR"))
    return OPENAI_API_KEY, main_dir, triplets_csv_path


@app.cell
def _(pd, triplets_csv_path):
    # load the csv:
    triplets_csv = pd.read_csv(triplets_csv_path)
    return (triplets_csv,)


@app.cell
def _(OPENAI_API_KEY, dspy):
    #configure dspy
    lm = dspy.LM('openai/gpt-4o-mini',api_key=OPENAI_API_KEY)
    dspy.configure(lm=lm)
    return


@app.cell
def _():
    from collections import defaultdict
    import math

    def compute_balanced_loops(selected_adj, base_loops=5):
        """
        Compute loop multiplier for each subclass to balance the total number of generated triplets.
        """
        subclass_sizes = {key: len(val) for key, val in selected_adj.items()}
        max_size = max(subclass_sizes.values())

        loop_counts = {}
        for key, size in subclass_sizes.items():
            if size == 0:
                loop_counts[key] = 0
            else:
                multiplier = max_size / size
                loop_counts[key] = math.ceil(base_loops * multiplier)
        return loop_counts
    return (compute_balanced_loops,)


@app.cell
def _(mo, triplets_csv):
    # get ui selectors
    subclass = mo.ui.multiselect(triplets_csv['subclass'].unique(), label="Select subclasses")
    number_of_loops = mo.ui.slider(start=0, stop=20, step=1, value=1, label="Number of loops")
    on = mo.ui.button(label="ON/OFF on change", value=False, on_click=lambda value: not value)

    mo.vstack([subclass, number_of_loops, on])
    return number_of_loops, on, subclass


@app.cell
def _(compute_balanced_loops, mo, number_of_loops, subclass, triplets_csv):
    # get adjectives triplets
    selected_adj = {}
    if len(subclass.value) > 1:
        for _subclass in subclass.value:
            selected_adj[_subclass] = triplets_csv[triplets_csv['subclass'] == _subclass][['adjective1', 'adjective2', 'negation_token']].values
    else:
        selected_adj[_subclass] = triplets_csv[triplets_csv['subclass'] == _subclass][['adjective1', 'adjective2', 'negation_token']].values


    loop_counts1 = compute_balanced_loops(selected_adj, base_loops=number_of_loops.value)

    totals = {k: len(v) * loop_counts1[k] * 3 for k, v in selected_adj.items()}
    tab = mo.ui.tabs(label="Preview of sentences generation:", tabs={
        "Total Generation": sum(totals.values()),
        "Contradictory": totals.get("contradictory_antonyms", 0),
        "Contrary": totals.get("contrary_antonyms", 0)
    })
    tab
    return (selected_adj,)


@app.cell
def _(dspy):
    class SentenceGeneration(dspy.Signature):
        """Generate a simple, natural-sounding sentence using a target adjective."""
        adjective = dspy.InputField()
        sentence = dspy.OutputField(desc="A natural-sounding sentence using the adjective")
    class GenerateSentence(dspy.Module):
        def __init__(self):
            super().__init__()
            self.program = dspy.Predict(signature=SentenceGeneration)

        def forward(self, adjective):
            return self.program(adjective=adjective)
    return (SentenceGeneration,)


@app.cell
def _():
    import re
    def safe_replace(sentence, target_word, replacement_word):
        """
        Replaces whole-word occurrences of `target_word` in `sentence` with `replacement_word`.
        Case-sensitive and punctuation-aware.
        """
        pattern = r'\b' + re.escape(target_word) + r'\b'
        return re.sub(pattern, replacement_word, sentence)
    return (safe_replace,)


@app.cell
def _(Path, SentenceGeneration, dspy, safe_replace, tqdm):
    import json
    def generate_and_save_triplets(selected_adj, loop_counts, output_root="output_sentences"):
        generator = dspy.Predict(signature=SentenceGeneration)  # or GenerateSentence()

        for subclass, triples in selected_adj.items():
            output_dir = Path(output_root) / subclass
            output_dir.mkdir(parents=True, exist_ok=True)

            data = []
            n_loops = loop_counts.get(subclass, 0)

            for loop_idx in range(n_loops):
                for adj1, adj2, neg in tqdm(triples, desc=f"[{subclass}] loop {loop_idx+1}/{n_loops}"):
                    try:
                        pred = generator(adjective=adj1)
                        base_sentence = pred.sentence.strip()

                        if adj1 not in base_sentence:
                            continue

                        neg_sentence = safe_replace(base_sentence, adj1, neg)
                        antonym_sentence = safe_replace(base_sentence, adj1, adj2)

                        data.append({
                            "adjective": adj1,
                            "negation": neg,
                            "antonym": adj2,
                            "adj_sentence": base_sentence,
                            "negated_sentence": neg_sentence,
                            "antonym_sentence": antonym_sentence,
                            "subclass": subclass
                        })

                    except Exception as e:
                        print(f"Error with '{adj1}' in subclass '{subclass}': {e}")
                        continue

            # Save to JSONL
            output_file = output_dir / "triplets.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for row in data:
                    json.dump(row, f, ensure_ascii=False)
                    f.write("\n")

    return generate_and_save_triplets, json


@app.cell
def _(
    compute_balanced_loops,
    generate_and_save_triplets,
    main_dir,
    number_of_loops,
    on,
    os,
    selected_adj,
):
    output_root = main_dir / "vae_generation"
    os.makedirs(output_root, exist_ok=True)

    if on.value:
        loop_counts = compute_balanced_loops(selected_adj, base_loops=number_of_loops.value)
        generate_and_save_triplets(selected_adj=selected_adj, output_root=output_root, loop_counts=loop_counts)
    return (output_root,)


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""# Embeddings Generation""")
    return


@app.cell
def _():
    from sentence_transformers import SentenceTransformer
    from typing import Dict, List
    return Dict, List, SentenceTransformer


@app.cell
def _(Dict, List, json):
    def load_triplets_json(filepath)->List:
        with open(filepath, "r", encoding='utf-8') as f:
            return [json.loads(line) for line in f]
        
    def normalize_not_adj(triplet:dict)->Dict:
        adj = triplet["adjective"]
        not_adj_sentence = triplet["negated_sentence"].replace(f"not_{adj}", f"not {adj}")
        return {
            "adj_sentence": triplet["adj_sentence"],
            "negated_sentence": not_adj_sentence,
            "antonym_sentence": triplet["antonym_sentence"],
            "subclass": triplet["subclass"]
        }
    return load_triplets_json, normalize_not_adj


@app.cell
def _(SentenceTransformer, tqdm):
    def generate_embeddings(triplets:dict, model_name:str="all-MiniLM-L6-v2"):
        model = SentenceTransformer(model_name)

        sentences = []
        for triplet in tqdm(triplets, desc="Embedding Generation"):
            sentences.extend([
                triplet["adj_sentence"],
                triplet["negated_sentence"],
                triplet["antonym_sentence"]
            ])
        embeddings = model.encode(sentences, convert_to_tensor=True)
        n = len(triplets)
        embeddings = embeddings.view(n, 3, -1)
        return embeddings
    return (generate_embeddings,)


@app.cell
def _(output_root):
    [f.name for f in  output_root.iterdir()]
    return


@app.cell
def _(Path, load_triplets_json, output_root):
    file_path_contradictory = Path(output_root / 'contradictory_antonyms'/ 'triplets.jsonl')
    raw_triplets_contradictory = load_triplets_json(file_path_contradictory)
    return (raw_triplets_contradictory,)


@app.cell
def _(normalize_not_adj, raw_triplets_contradictory):
    normalized_triplets_contradictory = [normalize_not_adj(_trip) for _triip in raw_triplets_contradictory]
    return (normalized_triplets_contradictory,)


@app.cell
def _(generate_embeddings, normalized_triplets_contradictory):
    embeddings = generate_embeddings(normalized_triplets_contradictory)
    return (embeddings,)


@app.cell
def _(embeddings):
    embeddings.shape
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _(mo):
    mo.md(r"""# Variational AutoEncoder Experiment """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
