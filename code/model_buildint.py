import marimo

__generated_with = "0.13.6"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import os
    from pathlib import Path
    from tqdm import tqdm
    import torch as t
    from sklearn.decomposition import PCA
    from gensim.models import Word2Vec
    import numpy as np
    from dotenv import load_dotenv
    import altair as alt
    import marimo as mo
    import polars as pol
    import matplotlib.pylab as plt
    from mofresh import anywidget
    return PCA, Path, Word2Vec, alt, load_dotenv, mo, os, pd, tqdm


@app.cell
def _(Path, load_dotenv, os):
    # get paths 
    load_dotenv(".env")
    main_dir = Path(os.getenv("MAIN_DIR"))
    models_dir = main_dir / "models"
    token_path = main_dir / "adjectives_antonyms_negation_tokens.csv"
    return models_dir, token_path


@app.cell
def _(Word2Vec, pd):
    def extract_valid_triplets(csv_path, model_path):
        model = Word2Vec.load(model_path)
        vocab = set(model.wv.key_to_index)

        df = pd.read_csv(csv_path)

        triplets = []
        for _, row in df.iterrows():
            adj = str(row["adjective1"]).strip().lower()
            antonym = str(row["adjective2"]).strip().lower()
            not_adj = str(row["negation_token"]).strip().lower().replace(" ", "_")  # 
            relation = row["subclass"].strip().lower()

            if not_adj in vocab and adj in vocab and antonym in vocab:
                triplets.append((not_adj, adj, antonym, relation))

        return triplets, model
    return (extract_valid_triplets,)


@app.cell
def _(extract_valid_triplets, models_dir, token_path):
    triplets_w, pretrained_model = extract_valid_triplets(token_path, str(models_dir / "E_pretrained.model"))
    return pretrained_model, triplets_w


@app.cell
def _(pd, pretrained_model, tqdm, triplets_w):
    rows = []
    for not_adj, adj, ant, rel in tqdm(triplets_w):
        for label, word in zip(['not_adj', 'adj', 'antonym'], [not_adj, adj, ant]):
            if word in pretrained_model.wv:
                vec = pretrained_model.wv[word]
                rows.append({'word': word, 'relation': rel, 'role': label, **{f'dim{i}': v for i, v in enumerate(vec)}})

    df = pd.DataFrame(rows)
    return (df,)


@app.cell
def _(PCA, df):
    vec_cols = [col for col in df.columns if col.startswith("dim")]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(df[vec_cols])
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    return


@app.cell
def _(df, mo):
    filter = mo.ui.radio(set(df['relation'].values))
    return (filter,)


@app.cell
def _(alt, df, filter, mo):
    chart = mo.ui.altair_chart(alt.Chart(df[df['relation']==filter.value]).mark_point().encode(
        x="x",
        y="y",
        color='role',
        tooltip=["word", "relation", "role"]
    ))
    return (chart,)


@app.cell
def _(chart, filter, mo):
    mo.vstack([filter, chart,  mo.ui.table(chart.value)])
    return


@app.cell
def _(mo):
    mo.md(text="#### calculate distance by relation")
    return


@app.cell
def _():
    from scipy.spatial.distance import cosine
    return (cosine,)


@app.cell
def _(cosine, pd, pretrained_model, triplets_w):
    def _(triplets, model):
        rows2 = []
        for not_adj, adj, antonym, relation in triplets:
            v_n = model.wv[not_adj]
            v_a = model.wv[adj]
            v_ant = model.wv[antonym]

            d_adj_not = cosine(v_a, v_n)
            d_adj_ant = cosine(v_a, v_ant)
            d_not_ant = cosine(v_n, v_ant)
            rows2.append({
                "relation": relation,
                "adj": adj,
                "not_adj": not_adj,
                "antonym": antonym,
                "d_adj_not": d_adj_not,
                "d_adj_ant": d_adj_ant,
                "d_not_ant": d_not_ant,})

        df_dist = pd.DataFrame(rows2)

        # Global averages
        return df_dist

    df_dist = _(triplets_w, pretrained_model)
    return (df_dist,)


@app.cell
def _(df_dist):
    global_avg = df_dist[["d_adj_not", "d_adj_ant", "d_not_ant"]].mean()
    by_relation = df_dist.groupby("relation")[["d_adj_not", "d_adj_ant", "d_not_ant"]].mean().reset_index()
    return (by_relation,)


@app.cell
def _(alt, by_relation, mo):
    bar_chart = mo.ui.altair_chart(alt.Chart(by_relation.melt(id_vars="relation")).mark_bar().encode(
        x="relation:N",
        y="value:Q",
        color="variable:N",
        tooltip=["relation", 'variable', 'value']
    ).properties(title="Average Cosine Distance by Relation"))
    return (bar_chart,)


@app.cell
def _(bar_chart):
    bar_chart
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
