import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pylab as plt
    import altair as alt
    from pathlib import Path
    from tqdm import tqdm
    import os 
    import numpy as np
    from mofresh import anywidget
    from dotenv import load_dotenv 
    import re
    from collections import Counter
    from nltk.tokenize import word_tokenize
    import nltk
    from sklearn.decomposition import PCA
    nltk.download('punkt_tab')
    return Counter, Path, load_dotenv, mo, os, pd, tqdm, word_tokenize


@app.cell
def _(mo):
    mo.md(text="## Overview of training data")
    return


@app.cell
def _(Path, load_dotenv, os):
    #get directories path
    load_dotenv(dotenv_path=Path(".env")) 

    review_dir = Path(os.getenv("DATA_DIR"))
    tokens_path = Path(os.getenv("TOKENS"))
    adj_neg_path = Path(os.getenv("ADJ_ANT"))
    return review_dir, tokens_path


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Review Data

    1. It contains: 50k reviews, split into 25k training and 25k test set. the distribution is balanced (pos/neg).
    2. max 30 reviews per movie
    3. no overlapping movie between test, train set

    #### File structure
    - train(post/neg), test(pos/neg)
    - Each review is stored as [id]_[rating].txt => 200_8.txt
    - Tokenize bag of words and .vocab
    """
    )
    return


@app.cell
def _(mo):
    mo.md(text="### Antonyms data")
    return


@app.cell
def _(pd, tokens_path):
    df_tokens = pd.read_csv(tokens_path)
    df_tokens.head()
    return (df_tokens,)


@app.cell
def _(mo):
    mo.md(r"""### Get frequency of words""")
    return


@app.cell
def _(Counter, review_dir):
    adj_freq = Counter()
    antonym_freq = Counter()
    negated_form_freq = Counter()

    training_path = review_dir / "train"
    return adj_freq, antonym_freq, negated_form_freq, training_path


@app.cell
def _(
    Path,
    adj_freq,
    adj_set,
    antonym_freq,
    antonym_set,
    neg_set,
    negated_form_freq,
    word_tokenize,
):
    def tokenize(text:str):
        return word_tokenize(text.lower())

    def analyze_text(filepath:Path):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        tokens = tokenize(text)

        for i, word in enumerate(tokens):
            if word in adj_set:
                adj_freq[word] += 1
            if word in antonym_set:
                antonym_freq[word] += 1
            if word == "not" and i + 1 < len(tokens):
                next_word = tokens[i+1]
                if next_word in adj_set or next_word in antonym_set:
                    negated_form_freq[f"not_{next_word}"] += 1
            if word in neg_set:
                negated_form_freq[word] += 1
    return (analyze_text,)


@app.cell
def _(df_tokens):
    adj_set = set(df_tokens["adjective1"])
    antonym_set = set(df_tokens["adjective2"])
    neg_set = set(df_tokens['negation_token'])
    return adj_set, antonym_set, neg_set


@app.cell
def _(analyze_text, tqdm, training_path):
    for label in ["pos", "neg"]:
        folder = training_path / label
        for fname in tqdm(folder.iterdir(), desc=f"Exploring {label} reviews"):
            analyze_text(folder / fname)
    return


@app.cell
def _(adj_freq, antonym_freq, df_tokens, negated_form_freq):
    summary = df_tokens.copy()
    summary["adj_freq"] = summary["adjective1"].str.lower().map(adj_freq)
    summary["antonym_freq"] = summary["adjective2"].str.lower().map(antonym_freq)
    summary["negated_freq"] = summary["negation_token"].str.lower().map(negated_form_freq)
    summary = summary.fillna(0).astype({"adj_freq": int, "antonym_freq": int, "negated_freq": int})
    return (summary,)


@app.cell
def _(mo, summary):
    selector = mo.ui.radio(summary['subclass'].unique(), label="Select the antonym class that you want to analyze")
    return (selector,)


@app.cell
def _(mo, selector, summary):
    mo.vstack([selector, summary[summary['subclass']==selector.value].sort_values(by=['frequency'], ascending=False)])
    return


@app.cell
def _(mo):
    mo.md(text="### Replace Negation adj with correct token:")
    return


@app.cell
def _(adj_set, antonym_set, summary):
    pos_set = adj_set.union(antonym_set)
    negation_set = set(summary["negation_token"])
    return (pos_set,)


@app.cell
def _(word_tokenize):
    def replace_negations(text, adj_set):
        tokens = word_tokenize(text.lower())
        new_tokens = []
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == "not" and tokens[i + 1] in adj_set:
                new_tokens.append(f"not_{tokens[i + 1]}")
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        if i == len(tokens) - 1:
            new_tokens.append(tokens[i])
        return " ".join(new_tokens)

    def process_file(src_path, dst_path, adj_set):
        with open(src_path, "r", encoding="utf-8") as f:
            text = f.read()
        new_text = replace_negations(text, adj_set)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(new_text)
    return (process_file,)


@app.cell
def _(OUT_DIR):
    post_train = OUT_DIR /'train'/ 'pos'
    len(list(post_train.glob("*.txt")))
    return


@app.cell
def _(os, pos_set, process_file, review_dir, tqdm):
    OUT_DIR = review_dir / "processed_imdb"
    os.makedirs(review_dir / "processed_imdb",exist_ok=True)

    for split in ["train", "test"]:
        for lab in ["pos", "neg"]:
            src_folder = review_dir / split / lab
            dst_folder = OUT_DIR / split / lab
            os.makedirs(dst_folder, exist_ok=True)
            if len(list(dst_folder.glob("*.txt"))) == 12500:
                print(f"{dst_folder} already processed during a previous run, skipping...")
                continue 
            else:
                for f in tqdm(src_folder.iterdir(), desc=f"Explring {split}, {lab}"):
                    src_path = src_folder  / f.name
                    dst_path = dst_folder / f.name
                    process_file(src_path, dst_path, pos_set)
    return (OUT_DIR,)


@app.cell
def _(mo):
    mo.md(text="### Model Trial")
    return


@app.cell
def _():
    from gensim.models import Word2Vec
    return (Word2Vec,)


@app.cell
def _(OUT_DIR, Path):
    PROCESSED_DIR = OUT_DIR / "train"
    augmented_dir = Path('/Users/thomasbush/Documents/Vault/CIMEC/LSC/seminar_paper/code/synthetic_data/reviews')
    SAVE_PATH = Path("/Users/thomasbush/Documents/Vault/CIMEC/LSC/seminar_paper/code/models/E_pretrained.model")
    return PROCESSED_DIR, SAVE_PATH, augmented_dir


@app.cell
def _(os, tqdm):
    from typing import List
    def load_tokenized_sentences(data_dir:List)->List:
        sentences = []
        for directory in data_dir:
            for label in ["pos", "neg"]:
                label_dir = directory / label
                for fname in tqdm(os.listdir(label_dir)):
                    if fname.endswith(".txt"):
                        fpath = label_dir / fname
                        with open(fpath, "r", encoding="utf-8") as f:
                            text = f.read()
                            tokens = text.split()  # already pre-tokenized
                            if len(tokens) > 0:
                                sentences.append(tokens)
        return sentences
    return (load_tokenized_sentences,)


@app.cell
def _(PROCESSED_DIR, augmented_dir, load_tokenized_sentences):
    list_of_dirs = [PROCESSED_DIR, augmented_dir]
    sent = load_tokenized_sentences(list_of_dirs)
    return (sent,)


@app.cell
def _(mo):
    train_model = mo.ui.button(label="Train model On/Off", value=False)
    train_model
    return (train_model,)


@app.cell
def _(Word2Vec, sent, train_model):
    if train_model.value:
        model = Word2Vec(
            sentences=sent,
            vector_size=300,
            window=10,
            min_count=3,
            sg=1,       # Skip-gram (better for rare words)
            workers=4   # Use available CPU cores
        )
    return (model,)


@app.cell
def _(Path, SAVE_PATH, model):
    Path("/Users/thomasbush/Documents/Vault/CIMEC/LSC/seminar_paper/code/models/").mkdir(exist_ok=True)
    model.save(str(SAVE_PATH))
    return


@app.cell
def _(model):
    #check if we have vecs:
    "not_much" in model.wv
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
