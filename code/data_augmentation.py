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
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY_PERSONAL"))
    dspy.configure(lm=lm)

    lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
    lm(
        messages=[{"role": "user", "content": "Say this is a test!"}]
    )  # => ['This is a test!']
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
    mo.md(r"""# Data Generation """)
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
    adj_csv[adj_csv["subclass"] == 'contrary_antonyms']
    return


@app.cell
def _(adj_csv):
    adj_type = "contradictory_antonyms"
    filtered = adj_csv[adj_csv['subclass'] == adj_type]
    return (filtered,)


@app.cell
def _(dspy):
    class MovieReviewSignature(dspy.Signature):
        adjective = dspy.InputField(desc="An adjective to include in the review")
        sentiment = dspy.InputField(desc="positive or negative")
        review = dspy.OutputField(desc="A short movie review using the adjective and matching the sentiment")
    return (MovieReviewSignature,)


@app.cell
def _(MovieReviewSignature, dspy, os, output_dir):
    os.makedirs(output_dir / "reviews/pos", exist_ok=True)
    os.makedirs(output_dir / "reviews/neg", exist_ok=True)
    rev_dir = output_dir / "reviews"

    # DSPy prediction setup
    predict = dspy.Predict(MovieReviewSignature)

    def generate_and_save_review(adjective: str, sentiment: str, file_id: int):
        result = predict(adjective=adjective, sentiment=sentiment)
        review_text = result.review.strip()

        folder = "pos" if sentiment == "positive" else "neg"
        file_path = rev_dir / folder / f'review_{file_id}.txt'

        with open(file_path, "w") as f:
            f.write(review_text)

        return review_text
    return (generate_and_save_review,)


@app.cell
def _():
    from tqdm import tqdm
    return (tqdm,)


@app.cell
def _(filtered, generate_and_save_review, tqdm):
    file_id = 0
    for column in ['adjective1', 'adjective2', 'negation_token']:
        for i, row in tqdm(filtered.iterrows(), total=len(filtered)):
            adj = row[column]
            sentiment = row[f"{column}_sentiment"]
            generate_and_save_review(str(adj), sentiment, file_id)
            file_id += 1
    return


@app.cell
def _(filtered):
    filtered.head()
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    return


if __name__ == "__main__":
    app.run()
