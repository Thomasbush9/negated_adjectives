import marimo

__generated_with = "0.13.6"
app = marimo.App(width="columns")


@app.cell(column=0)
def _(mo):
    mo.md(r"""# Imports and Description of the Notebook""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd 
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Goal of the Notebook🥇

    1. Use dspy to create synthetic data for the project
    2. Validate the synthetic data
    3. Save it in the right format

    ### Strategy

    We are going to use dspy to generate a dataset containing the triplets that we want to use for the negated adjective study inside movie reviews similar to our current dataset. To validate the strategy we are going to train a logistic regression model over the original dataset and run an inference over the new reviews to see if they are similar in the information conveyed to the real data. 

    ### Possible Additions 

    1. Add some kind of optimization for dspy
    2.
    """
    )
    return


@app.cell
def _():
    from dotenv import load_dotenv
    import os

    load_dotenv(dotenv_path=".env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TOGETHER_API_KEY = os.getenv("TOGETHER_AI_API")
    return load_dotenv, os


@app.cell
def _(load_dotenv, os):
    import dspy

    load_dotenv()
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY_PERSONAL"), temperature=0.7)
    dspy.configure(lm=lm)


    return (dspy,)


@app.cell
def _():
    from pathlib import Path
    return (Path,)


@app.cell
def _():
    return


@app.cell(column=1)
def _(mo):
    mo.md(r"""# Data Generation""")
    return


@app.cell
def _(os):
    adj_csv_path = os.getenv("TOKENS")
    main_dir = os.getenv("MAIN_DIR")
    return (main_dir,)


@app.cell
def _(os, pd):
    adj_csv = pd.read_csv(os.getenv("TOKENS"))
    return (adj_csv,)


@app.cell
def _(adj_csv):
    adj_csv[adj_csv['subclass'] == 'contradictory_antonyms']['negation_token'].values
    return


@app.cell
def _(Path, main_dir, os):
    #generate output dir
    output_dir = Path(main_dir) / 'synthetic_data'
    os.makedirs(output_dir, exist_ok=True)
    return (output_dir,)


@app.cell
def _():
    # generate simple input data for our generation: contradictory ant
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(word:str):
        score = analyzer.polarity_scores(word)["compound"]
        return "positive" if score >= 0 else "negative"
    return (analyzer,)


@app.cell
def _(random):
    def label_from_score(score, flip_probability=0.2):
        label = "positive" if score >= 0 else "negative"
        if random.random() < flip_probability:
            return "negative" if label == "positive" else "positive"
        return label
    return (label_from_score,)


@app.cell
def _():
    import random
    return (random,)


@app.cell
def _(adj_csv, analyzer, label_from_score):
    # add a sub category for each word about the sentiment
    for col in ['adjective1', 'adjective2', 'negation_token']:
        # Get sentiment score
        scores = adj_csv[col].apply(lambda x: analyzer.polarity_scores(str(x))["compound"])
        adj_csv[f"{col}_score"] = scores

        # Add sub-category sentiment label (with some randomness)
        adj_csv[f"{col}_sentiment"] = scores.apply(lambda s: label_from_score(s, flip_probability=0.2))

    return


@app.cell
def _(adj_csv):
    adj_csv[adj_csv["subclass"] == 'contradictory_antonyms']
    return


@app.cell
def _(adj_csv):
    adj_csv
    return


@app.cell
def _(dspy):
    class MovieReviewSignature(dspy.Signature):
        adjective = dspy.InputField(desc="An adjective to include in the review")
        sentiment = dspy.InputField(desc="positive or negative")
        review = dspy.OutputField(desc="A detailed movie review of at least 6-8 sentences that uses the adjective and matches the sentiment.")
    return (MovieReviewSignature,)


@app.cell
def _(MovieReviewSignature, dspy, output_dir):
    rev_dir = output_dir / "reviews"
    pos_dir = rev_dir / "pos"
    neg_dir = rev_dir / "neg"

    # Make sure directories exist
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # DSPy prediction setup
    predict = dspy.Predict(MovieReviewSignature)

    def generate_and_save_review(adjective: str, sentiment: str, file_id: int):
        result = predict(adjective=adjective, sentiment=sentiment)
        review_text = result.review.strip()

        folder = "pos" if sentiment == "positive" else "neg"
        dir_path = rev_dir / folder
        file_path = dir_path / f'review_{file_id}.txt'

        with open(file_path, "w") as f:
            f.write(review_text)

        return review_text
    return generate_and_save_review, neg_dir, pos_dir


@app.cell
def _():
    from tqdm import tqdm
    return (tqdm,)


@app.cell(hide_code=True)
def _(adj_csv, mo):
    button = mo.ui.button(
        value=False, 
        on_click=lambda value: True if not value else False, label='dspy run on/off'
    )
    adj_type = mo.ui.radio(adj_csv['subclass'].unique(), value='contrary_antonyms')
    text = mo.ui.slider(start=1, stop=10, step=1, value=1, show_value=True, label='Number of loops')

    mo.hstack([adj_type, mo.vstack([button, text], align='stretch')], align='center')
    return adj_type, button, text


@app.cell(hide_code=True)
def _(adj_csv, adj_type, text):
    filtered = adj_csv[adj_csv['subclass'] == adj_type.value]
    tot_loops = text.value * len(filtered) * 3
    print(f"You are about to create {tot_loops} reviews, are you sure to start it?")
    return (filtered,)


@app.cell
def _(
    button,
    filtered,
    generate_and_save_review,
    neg_dir,
    pos_dir,
    text,
    tqdm,
):
    existing_pos = len(list(pos_dir.glob("review_*.txt")))
    existing_neg = len(list(neg_dir.glob("review_*.txt")))
    file_id = existing_pos + existing_neg
    if button.value:
        for n in range(0, text.value):
            for column in ['adjective1', 'adjective2', 'negation_token']:
                for i, row in tqdm(filtered.iterrows(), total=len(filtered)):
                    adj = row[column]
                    sentiment = row[f"{column}_sentiment"]
                    generate_and_save_review(str(adj), sentiment, file_id)
                    file_id += 1
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _(mo):
    mo.md(r"""## Results Evaluation""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Qualitative Evaluaiton

    We are going to review and compute some statistics to compare the real dataset and the synthetic one
    """
    )
    return


@app.cell
def _(os, pd, tqdm):
    #load the reviews: 
    def load_reviews_from_folder(base_path):
        rows = []
        for label in ['pos', 'neg']:
            folder = os.path.join(base_path, label)
            for filename in tqdm(os.listdir(folder), desc=f'extracting {label} reviews'):
                if filename.endswith(".txt"):
                    path = os.path.join(folder, filename)
                    with open(path, 'r') as f:
                        text = f.read().strip()
                    rows.append({
                        "filename": filename,
                        "review": text,
                        "label": label
                    })
        return pd.DataFrame(rows)
    return (load_reviews_from_folder,)


@app.cell
def _(Path, main_dir, output_dir):
    real_data_path = Path(main_dir) / 'aclImdb/train'
    synthetic_data_path = output_dir / 'reviews'
    return real_data_path, synthetic_data_path


@app.cell
def _(load_reviews_from_folder, real_data_path, synthetic_data_path):
    real_df = load_reviews_from_folder(base_path=real_data_path)
    synthetic_df = load_reviews_from_folder(base_path=synthetic_data_path)
    return real_df, synthetic_df


@app.cell
def _(synthetic_df):
    synthetic_df.head()
    return


@app.cell
def _():
    # show some examples:
    def show_examples(df, label_col='label', n=3):
        print("\n Example Reviews")
        for label in df[label_col].unique():
            print(f"\n--- {label.upper()} ---")
            examples = df[df[label_col] == label].sample(n)
            for i, row in examples.iterrows():
                print(f"{row['filename']}: {row['review'][:200]}...\n")

    def compute_length_stats(df, name="Dataset"):
        df["word_count"] = df["review"].apply(lambda x: len(x.split()))
        print(f"\n📊 {name} Stats:")
        print(f"Average length: {df['word_count'].mean():.2f} words")
        print(f"Shortest: {df['word_count'].min()} words")
        print(f"Longest: {df['word_count'].max()} words")
    return (compute_length_stats,)


@app.cell
def _(compute_length_stats, real_df, synthetic_df):
    compute_length_stats(real_df, "Real Reviews")
    compute_length_stats(synthetic_df, "Synthetic Reviews")
    return


@app.cell
def _(mo):
    mo.md(r"""### Sentiment Analysis""")
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.utils import shuffle
    import numpy as np
    return (
        LogisticRegression,
        TfidfVectorizer,
        classification_report,
        np,
        shuffle,
    )


@app.cell
def _(TfidfVectorizer, np, real_df, shuffle):
    vectorizer = TfidfVectorizer(max_features=5000)
    real_df_shuffle = shuffle(real_df, random_state=42)
    X_train = vectorizer.fit_transform(real_df_shuffle["review"])
    y_train = np.array(real_df_shuffle["label"].values)
    y_train = np.where(y_train == 'pos', 1, 0)
    return X_train, vectorizer, y_train


@app.cell
def _(X_train):
    X_train.shape
    return


@app.cell
def _(LogisticRegression, X_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    return (clf,)


@app.cell
def _(np, synthetic_df, vectorizer):
    X_test = vectorizer.transform(synthetic_df["review"])
    y_test = np.array(synthetic_df["label"].values)
    y_test = np.where(y_test == 'pos', 1, 0)
    return X_test, y_test


@app.cell
def _(X_test):
    X_test.shape
    return


@app.cell
def _(X_test, classification_report, clf, y_test):
    y_pred = clf.predict(X_test)

    print("\n📊 Classification Report on Synthetic Data:")
    print(classification_report(y_test, y_pred))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
