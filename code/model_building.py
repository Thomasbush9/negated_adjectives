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
    return PCA, Path, Word2Vec, alt, load_dotenv, mo, np, os, pd, t, tqdm


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
def _(mo):
    mo.md(text="### Model building")
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(mo):
    mo.md(text="So, we have the df with 300 dims one for each column, from that we can build the data for our custom NN to push the embeddings a bit")
    return


@app.cell
def _(triplets_w):
    # prepare data
    relation_types = sorted(list(set(row[3] for row in triplets_w)))
    relation_to_idx = {r:i for i, r in enumerate(relation_types)}
    num_classes = len(relation_types)
    return num_classes, relation_to_idx


@app.cell
def _(np, num_classes, pretrained_model, relation_to_idx, t, tqdm, triplets_w):
    # build embedding and label tensors
    embedding_list = []
    label_list = []

    for not_ad, ad, anto, rels in tqdm(triplets_w, desc="generating tensor"):
        try:
            vec_negation = pretrained_model.wv[not_ad]
            vec_adjective = pretrained_model.wv[ad]
            vec_antonym = pretrained_model.wv[anto]
        except KeyError:
            continue 
        trilet_vec = np.stack([vec_adjective, vec_antonym, vec_negation])
        embedding_list.append(trilet_vec)

        # one hot label
        lab = np.zeros(num_classes, dtype=np.float32)
        lab[relation_to_idx[rels]] = 1.0
        label_list.append(lab)

    # convert to tensor
    embedding_tensor = t.tensor(np.stack(embedding_list), dtype=t.float32) # (N, 3, 300)
    label_tensor = t.tensor(np.stack(label_list), dtype=t.float32) #(N, num_classes)

    print(f'Final tensor shape:', embedding_tensor.shape)
    print('finale label shape', label_tensor.shape)
    return


@app.cell
def _(t):
    # create a simple model: it should only operate on the negated adj, but remain kind of similar to the initial one 
    from torch import Tensor

    class NN(t.nn.Module):
        def __init__(self, num_hidden_features):
            super().__init__()
            self.layer1 = t.nn.Linear(in_features=300, out_features=num_hidden_features)
            self.act1 = t.nn.ReLU()

            self.layer2 = t.nn.Linear(num_hidden_features, 300)
            self.act2 = t.nn.ReLU()

        def forward(self, x:Tensor)->Tensor:
            assert len(x.shape) == 2, "Only one dim embedding supported"
            out = self.act2(self.layer2(self.act1(self.layer1(x))))
            return x + out #residual connection 


    return (Tensor,)


@app.cell
def _(Tensor, cosine):
    # define possible training loss:
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    def cosine_tensor(a:Tensor, b:Tensor):
        return F.cosine_similarity(a, b, dim=-1)

    def triplet_loss(relation:Tensor, adj: Tensor, ant:Tensor, pred:Tensor):
        sim_adj = cosine(pred, adj)    # similarity to adjective
        sim_ant = cosine(pred, ant)    # similarity to antonym

        # Encourage sim(pred, adj) > sim(pred, ant) for 'contrary'
        l1 = F.relu(sim_ant - sim_adj)

        # Encourage sim(pred, ant) > sim(pred, adj) for 'contradictory'
        l2 = F.relu(sim_adj - sim_ant)

        # One-hot relation: [contrary, contradictory]
        loss = relation[:, 0] * l1 + relation[:, 1] * l2
        return loss.mean()
    
    return


@app.cell
def _(relation_to_idx):
    relation_to_idx
    return


if __name__ == "__main__":
    app.run()
