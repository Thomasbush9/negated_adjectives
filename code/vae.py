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
    return DataLoader, F, Path, dspy, load_dotenv, mo, os, pd, torch, tqdm


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

    class SentenceWithAdjective(dspy.Signature):
        """Generate a natural-sounding sentence using the target adjective in a descriptive context."""
        adjective = dspy.InputField()
        sentence = dspy.OutputField(desc="A fluent sentence that uses the adjective meaningfully.")

    class GenerateBaseSentence(dspy.Module):
        def __init__(self):
            super().__init__()
            self.generator = dspy.Predict(signature=SentenceWithAdjective)

        def forward(self, adjective):
            return self.generator(adjective=adjective)
    class RewriteSentenceWithAdjectiveChange(dspy.Signature):
        """
        Rewrite a sentence that contains an adjective by replacing it with a new one.
        The rewritten sentence should preserve the structure, grammar, and meaning.
        """
        original_sentence = dspy.InputField()
        old_adjective = dspy.InputField()
        new_adjective = dspy.InputField()
        rewritten_sentence = dspy.OutputField(desc="Fluent and meaningful rewritten sentence.")
    class RewriteWithAdjective(dspy.Module):
        def __init__(self):
            super().__init__()
            self.rewriter = dspy.Predict(signature=RewriteSentenceWithAdjectiveChange)

        def forward(self, original_sentence, old_adjective, new_adjective):
            return self.rewriter(
                original_sentence=original_sentence,
                old_adjective=old_adjective,
                new_adjective=new_adjective
            )

    return GenerateBaseSentence, RewriteWithAdjective


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
    return


@app.cell
def _(GenerateBaseSentence, Path, RewriteWithAdjective, tqdm):
    import json
    def generate_and_save_triplets(selected_adj, loop_counts, output_root="output_sentences"):
        base_generator = GenerateBaseSentence()
        rewriter = RewriteWithAdjective()

        for subclass, triples in selected_adj.items():
            output_dir = Path(output_root) / subclass
            output_dir.mkdir(parents=True, exist_ok=True)

            data = []
            n_loops = loop_counts.get(subclass, 0)

            for loop_idx in range(n_loops):
                for adj1, adj2, neg in tqdm(triples, desc=f"[{subclass}] loop {loop_idx+1}/{n_loops}"):
                    try:
                        # Step 1: Generate base sentence with original adjective
                        base_result = base_generator(adjective=adj1)
                        base_sentence = base_result.sentence.strip()

                        # Ensure the original adjective is in the sentence
                        if adj1 not in base_sentence:
                            continue

                        # Step 2: Rewrite with negation and antonym using old_adjective
                        neg_result = rewriter(original_sentence=base_sentence, old_adjective=adj1, new_adjective=neg)
                        ant_result = rewriter(original_sentence=base_sentence, old_adjective=adj1, new_adjective=adj2)

                        data.append({
                            "adjective": adj1,
                            "negation": neg,
                            "antonym": adj2,
                            "adj_sentence": base_sentence,
                            "negated_sentence": neg_result.rewritten_sentence.strip(),
                            "antonym_sentence": ant_result.rewritten_sentence.strip(),
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
    output_root = main_dir / "vae_generation_improved"
    os.makedirs(output_root, exist_ok=True)

    if on.value:
        loop_counts = compute_balanced_loops(selected_adj, base_loops=number_of_loops.value)
        generate_and_save_triplets(selected_adj=selected_adj, output_root=output_root, loop_counts=loop_counts)
    return (output_root,)


@app.cell
def _(output_root):
    output_root
    return


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
def _(mo):
    mo.md(
        r"""
    ### Model Definition

    Specifics of the architecture

    1. Encoder: 
    """
    )
    return


@app.cell
def _():
    import torch.nn as nn
    from torch import Tensor
    import torch as t
    from torch.utils.data import random_split
    return Tensor, nn, random_split, t


@app.cell
def _(Tensor, nn, t):
    class SentenceVAE(nn.Module):
        def __init__(self, input_dim:int=384, latent_dim:int=32, hidden_dim:int=128):
            super(SentenceVAE, self).__init__()
            self.input_dim = input_dim
            self.latent = latent_dim
            self.hidden = hidden_dim

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                nn.Linear(hidden_dim*2, input_dim)
            )
        def encode(self, x:Tensor):
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        def reparameterize(self, mu:Tensor, logvar:Tensor)->Tensor:
            std = t.exp(.5*logvar)
            eps = t.rand_like(std)
            return mu + eps * std
        def decode(self, z:Tensor)->Tensor:
            assert z.shape[-1] == self.latent, f'Input decoder must be: {self.latent}'
            return self.decoder(z)
        def forward(self, x:Tensor):
            assert x.shape[-1] == self.input_dim, f'Input dimensions must be {self.input_dim}'
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            return x_recon, mu, logvar
    return (SentenceVAE,)


@app.cell
def _(F, Tensor, t):
    #loss func
    def vae_loss(recon_x:Tensor, x:Tensor, mu:Tensor, logvar:Tensor):
        '''Simple reconstruction loss + Kernel Div loss'''
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -.5 * t.sum(1+logvar - mu.pow(2)-logvar.exp()) / x.shape[0]
        return recon_loss + kl_loss, recon_loss, kl_loss
    return (vae_loss,)


@app.cell
def _():
    from dataclasses import dataclass
    from typing import Literal
    from torch.utils.data import TensorDataset
    return Literal, TensorDataset, dataclass


@app.cell
def _(AutoencoderArgs, Literal):

    class AutoencoderTrainer:
        def __init__(self, args: AutoencoderArgs, device:Literal['cpu', 'mps']):
            self.args = args
 
    return


@app.cell
def _(SentenceVAE, TensorDataset, dataclass):
    @dataclass
    class SentenceVAE_TrainingArgs:
        dataset:TensorDataset
        vae:SentenceVAE
        latent:int=64
        hidden:int=256
        lr=1e-3
        epochs:int=20
        batch_size:int=64
        use_wandb:bool=False
        wandb_project:str='LSC'
        wandn_name:str='Sentence VAE'
    
    return (SentenceVAE_TrainingArgs,)


@app.cell
def _(
    DataLoader,
    Literal,
    SentenceVAE_TrainingArgs,
    Tensor,
    random_split,
    t,
    torch,
    tqdm,
    vae_loss,
    wandb,
):
    #TODO: add different loss func here
    class SentenceVAE_Trainer:
        def __init__(self, args: SentenceVAE_TrainingArgs, device: Literal['cpu', 'mps']):
            self.args = args
            self.device = device
            self.beta = 0.1
            self.warmup_epochs = 5

            # Split into train/val sets
            total_len = len(args.dataset)
            val_len = int(0.1 * total_len)
            train_len = total_len - val_len
            self.trainset, self.valset = random_split(args.dataset, [train_len, val_len])

            self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
            self.valloader = DataLoader(self.valset, batch_size=args.batch_size, shuffle=False)

            self.model = args.vae().to(self.device)
            self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr)
            self.scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )

            self.step = 0
            self.loss_fn = vae_loss

        def training_step(self, x: Tensor) -> tuple:
            self.model.train()
            pred, mu, logvar = self.model.forward(x)
            _, loss_rec, loss_kl = self.loss_fn(x=x, recon_x=pred, mu=mu, logvar=logvar)
            loss = loss_rec + self.beta * loss_kl
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.step += x.shape[0]
        

            if self.args.use_wandb:
                wandb.log({"loss": loss.item(), "loss_rec": loss_rec.item(), "loss_kl": loss_kl.item()}, step=self.step)

            return loss.item(), loss_rec.item(), loss_kl.item()

        def evaluate(self) -> float:
            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in self.valloader:
                    x = batch[0].to(self.device)
                    pred, mu, logvar = self.model.forward(x)
                    loss, _, _ = self.loss_fn(x=x, recon_x=pred, mu=mu, logvar=logvar)
                    total_loss += loss.item() * x.size(0)
            return total_loss / len(self.valset)

        def train(self):
            self.step = 0
            if self.args.use_wandb:
                wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
                wandb.watch(self.model)

            best_val_loss = float('inf')
            for epoch in range(self.args.epochs):
                progress_bar = tqdm(self.trainloader, total=len(self.trainloader), ascii=True)
            
                for batch in progress_bar:
                    x = batch[0].to(self.device)
                    loss, loss_rec, loss_kl = self.training_step(x)
                    progress_bar.set_description(f"{epoch=:02d}, {loss=:.4f}, step={self.step:05d}, kl loss:{loss_kl:.4f}")
                # self.beta = min(1.0, epoch / self.warmup_epochs)
                val_loss = self.evaluate()
                self.scheduler.step(val_loss)
                print(f"Validation loss after epoch {epoch}: {val_loss:.4f}")

            if self.args.use_wandb:
                wandb.finish()
            return self.model, val_loss

    return (SentenceVAE_Trainer,)


@app.cell
def _(
    SentenceVAE,
    SentenceVAE_Trainer,
    SentenceVAE_TrainingArgs,
    TensorDataset,
    embeddings,
    torch,
):
    embeddings_tensor_negation = embeddings[:, 1, :]
    embeddings_all = torch.cat([
        embeddings[:, 0, :],  # adj
        embeddings[:, 1, :],  # not_adj
        embeddings[:, 2, :]   # antonym
    ], dim=0)
    args = SentenceVAE_TrainingArgs(
        dataset=TensorDataset(embeddings_all),
        vae=SentenceVAE,
        batch_size=64, 
        epochs=20,
        use_wandb=False
    )

    trainer = SentenceVAE_Trainer(args=args, device='mps')
    model_vae_neg, loss_fin = trainer.train()
    return embeddings_tensor_negation, model_vae_neg


@app.cell
def _(embeddings_tensor_negation):
    print(embeddings_tensor_negation.mean(), embeddings_tensor_negation.std())
    return


@app.cell
def _(mo):
    mo.md(r"""### Check results of Latent space""")
    return


@app.cell
def _(embeddings, model_vae_neg, torch):
    model_vae_neg.eval()
    with torch.no_grad():
        z_adj = model_vae_neg.encode(embeddings[:, 0, :])[0]      # shape: (n, latent_dim)
        z_not = model_vae_neg.encode(embeddings[:, 1, :])[0]
        z_ant = model_vae_neg.encode(embeddings[:, 2, :])[0]

    return z_adj, z_ant, z_not


@app.cell
def _(z_not):
    z_not.shape
    return


@app.cell
def _(F, z_adj, z_not, z_pred):
    z_adj_norm = F.normalize(z_adj, dim=1)
    z_not_norm = F.normalize(z_not, dim=1)
    z_pred_norm = F.normalize(z_pred, dim=1)

    return


@app.cell
def _(z_adj, z_not):
    neg_vectors = z_not.mean(dim=0) - z_adj.mean(dim=0)  # shape: (n, latent_dim)
    # negation_vector = neg_vectors.mean(dim=0, keepdim=True)  # shape: (1, latent_dim)

    return (neg_vectors,)


@app.cell
def _(neg_vectors):
    neg_vectors.shape
    return


@app.cell
def _(F, neg_vectors, z_adj, z_ant, z_not):
    # Compute predicted negated vectors
    z_pred = z_adj + neg_vectors  # apply the learned negation vector

    # Cosine similarity
    sim_pred_not = F.cosine_similarity(z_pred, z_not, dim=1).mean().item()
    sim_pred_ant = F.cosine_similarity(z_pred, z_ant, dim=1).mean().item()

    print(f"Mean cosine similarity (predicted vs. actual negation): {sim_pred_not:.4f}")
    print(f"Mean cosine similarity (predicted vs. antonym): {sim_pred_ant:.4f}")

    return (z_pred,)


@app.cell
def _(torch, z_adj, z_ant, z_not, z_pred):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    all_z = torch.cat([z_adj, z_not, z_ant, z_pred], dim=0).cpu().numpy()
    labels = (["adj"] * len(z_adj)) + (["not"] * len(z_not)) + (["ant"] * len(z_ant)) + (["pred"] * len(z_pred))

    # Reduce to 2D
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(all_z)

    # Plot
    plt.figure(figsize=(8, 6))
    for label in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(z_pca[idxs, 0], z_pca[idxs, 1], label=label, alpha=0.6)

    plt.legend()
    plt.title("Latent space PCA projection of adjective, negation, antonym and predicted negation vectors")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(F, z_ant, z_pred):
    cos_sim_to_ant = F.cosine_similarity(z_pred, z_ant, dim=1)
    print(f"Mean cosine similarity (z_pred vs z_antonym): {cos_sim_to_ant.mean().item():.4f}")

    return (cos_sim_to_ant,)


@app.cell
def _(cos_sim, cos_sim_to_ant):
    delta = cos_sim_to_ant - cos_sim
    print(f"Mean (pred closer to antonym than actual not_adj): {delta.mean().item():.4f}")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
