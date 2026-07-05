#ML Pipeline
# Uses the Zenodo dataset

#1. - Loads the dataset
#2. - Merge text and labels together
#3. - Cleans and pre-processes the tickets text NLTK - This step transforms raw support ticket text into a structured, standardized format using the Natural Language Toolkit (NLTK). It improves the accuracy of downstream machine learning models by stripping away noise.
#4. - Converts text into numbers using TF-IDF
#5.- train 4 different ML classifiers
#6. Evaluates each one and picks the best
#7. Saves the best model to the models folder

import os  # for building file paths
import re  # for cleaning text with regex
import string  # for removing punctuation
import warnings  # to suppress unnecessary warnings

import joblib
# NLTK — Natural Language Toolkit (used for text cleaning)
import nltk
import numpy as np
import pandas as pd  # for loading and working with CSV files
from nltk.corpus import stopwords  # common words to remove (the, is, a...)
from nltk.stem import WordNetLemmatizer  # reduces words to root form (running→run)
from sklearn.ensemble import RandomForestClassifier
# Scikit-learn — the ML library
from sklearn.feature_extraction.text import TfidfVectorizer  # converts text to numbers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,  # detailed per-class metrics
    confusion_matrix,  # shows where model gets confused
    accuracy_score,  # overall % correct
    f1_score,  # balanced metric (handles uneven class sizes)
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline  # chains steps together
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

# ── Download NLTK data (runs once, then uses cached version) ───
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))  # set of common words to ignore
LEMMATIZER = WordNetLemmatizer()               # tool to reduce words to root form

# ── File Paths ─────────────────────────────────────────────────
# __file__ = this script's location (src/train_models.py)
# dirname goes up one level to get the project root
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")    # where the CSVs are
MODEL_DIR = os.path.join(ROOT, "models")  # where we save trained models
os.makedirs(MODEL_DIR, exist_ok=True)     # create models/ if it doesn't exist


# ══════════════════════════════════════════════════════════════
# STEP 1 — TEXT CLEANING
# ══════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Takes raw messy ticket text and cleans it for ML.

    Example:
      Input:  "File Share Access - [TICKET ID] - John needs access!"
      Output: "file share access john need access"
    """
    text = str(text).lower()                                    # UPPERCASE → lowercase

    text = re.sub(r"\[.*?\]", " ", text)                       # remove [NAME], [TICKET ID]
    text = re.sub(r"http\S+|www\.\S+", " ", text)              # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                       # remove email addresses
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", " ", text)                           # remove numbers

    # Split into individual words, remove stopwords, lemmatize each word
    tokens = [
        LEMMATIZER.lemmatize(tok)        # e.g. "running" → "run"
        for tok in text.split()
        if tok not in STOP_WORDS         # skip words like "the", "is", "and"
        and len(tok) > 2                 # skip very short words
    ]

    return " ".join(tokens)   # join back into one string


# ══════════════════════════════════════════════════════════════
# STEP 2 — BUILD ML PIPELINE
# ══════════════════════════════════════════════════════════════

def build_pipeline(classifier) -> Pipeline:
    """
    Creates a Pipeline that chains two steps:
      Step 1 — TF-IDF: converts text into numbers
      Step 2 — Classifier: learns patterns from those numbers

    Why a Pipeline? So we can train both steps together
    with one single .fit() call.

    TF-IDF settings explained:
      ngram_range=(1,2)  → uses single words AND pairs of words
                           e.g. "file", "share", "file share"
      max_features=15000 → only keep the 15,000 most useful words
      sublinear_tf=True  → reduces the dominance of very common words
      min_df=2           → ignore words that appear in only 1 ticket
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=15_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", classifier),
    ])


# ══════════════════════════════════════════════════════════════
# STEP 3 — EVALUATE A TRAINED MODEL
# ══════════════════════════════════════════════════════════════

def evaluate_model(name, pipeline, X_test, y_test, label_names) -> dict:
    """
    Runs the trained model on the test set and calculates metrics.

    Returns a dictionary with:
      - accuracy: overall % correct
      - f1_weighted: balanced score (our main metric for selection)
      - confusion_matrix: grid showing where errors happen
      - report: full per-class breakdown
    """
    y_pred = pipeline.predict(X_test)   # make predictions on unseen test data

    acc  = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1_w:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names))

    return {
        "name":             name,
        "accuracy":         round(acc, 4),
        "f1_weighted":      round(f1_w, 4),
        "report":           classification_report(y_test, y_pred, target_names=label_names, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "label_names":      label_names,
        "y_test":           list(y_test),
        "y_pred":           list(y_pred),
    }


# ══════════════════════════════════════════════════════════════
# STEP 4 — MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════

def train_and_save():
    """
    Runs the full ML pipeline from raw CSVs to saved model files.
    Call this from the bottom of the script.
    """

    print("Loading dataset...")
    X_train_df = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test_df = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))  # ticket text (test)
    y_train_df = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))  # labels (train)
    y_test_df = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))  # labels (test)

    # Merge text + labels together using the shared 'id' column
    # This is like a VLOOKUP in Excel — matches rows by ID
    train_df = X_train_df.merge(y_train_df, on="id")
    test_df = X_test_df.merge(y_test_df, on="id")

    print(f"  Train tickets: {len(train_df)}")
    print(f"  Test tickets:  {len(test_df)}")
    print(f"  Categories:    {sorted(train_df['category_truth'].unique())}")

    # ── Clean all ticket text ──────────────────────────────────
    print("\nCleaning text ...")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    # Remove any rows where cleaning left an empty string
    train_df = train_df[train_df["clean_text"].str.strip() != ""].reset_index(drop=True)
    test_df = test_df[test_df["clean_text"].str.strip() != ""].reset_index(drop=True)

    # ── Encode category labels as numbers ──────────────────────
    # ML models need numbers, not text labels
    # e.g. "Fileservice" → 2, "O365" → 4
    encoder = LabelEncoder()
    all_cats = pd.concat([train_df["category_truth"], test_df["category_truth"]])
    encoder.fit(all_cats)  # learn all possible categories

    y_train = encoder.transform(train_df["category_truth"])  # convert train labels
    y_test = encoder.transform(test_df["category_truth"])  # convert test labels
    labels = list(encoder.classes_)  # e.g. ['Active Directory', 'EOL', ...]

    # Convert text to plain numpy arrays (required by sklearn)
    X_train = np.array(train_df["clean_text"].tolist())
    X_test = np.array(test_df["clean_text"].tolist())

    #Define classifiers
    classifiers = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, C=0.1, solver ="lbfgs"),
        "Linear Support Vector Machine" : LinearSVC(C=1.0, max_iter=2000),
        "Random Forest" : RandomForestClassifier(n_estimators=200, random_state= 42, n_jobs = -1),
        "Naives Bayes" : MultinomialNB(alpha=0.1)
    }

    # ── Train and evaluate each classifier ────────────────────
    print("\n" + "═" * 55)
    print("  TRAINING — 7-Class IT Support Ticket Classification")
    print("═" * 55)

    all_results = []
    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name} ...")
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)  # ← THIS is where learning happens
        result = evaluate_model(clf_name, pipe, X_test, y_test, labels)
        result["pipeline"] = pipe  # store the trained pipeline
        all_results.append(result)

    # ── Pick the best model by F1 score ───────────────────────
    best = max(all_results, key=lambda r: r["f1_weighted"])
    print(f"\n✓ Best model: {best['name']}  (F1 = {best['f1_weighted']:.4f})")

    # ── Add word count column for dashboard analytics ──────────
    train_df["word_count"] = train_df["text"].apply(lambda t: len(str(t).split()))
    test_df["word_count"] = test_df["text"].apply(lambda t: len(str(t).split()))

    # Combine train + test into one full dataset for the dashboard
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = full_df.rename(columns={"category_truth": "category"})

    # ── Save everything to the models/ folder ─────────────────
    print("\nSaving models ...")

    # Save the best trained model
    joblib.dump(best["pipeline"], os.path.join(MODEL_DIR, "best_category_model.pkl"))

    # Save the encoder so app.py can convert numbers back to category names
    joblib.dump(encoder, os.path.join(MODEL_DIR, "category_encoder.pkl"))

    # Strip the pipeline objects before saving results (not serialisable)
    def strip_pipeline(results):
        return [{k: v for k, v in r.items() if k != "pipeline"} for r in results]

    # Save all results and stats so the dashboard can read them
    results_payload = {
        "category": {
            "results": strip_pipeline(all_results),
            "best_model_name": best["name"],
            "labels": labels,
        },
        "dataset_stats": {
            "total": len(full_df),
            "train": len(train_df),
            "test": len(test_df),
            "category_dist": full_df["category"].value_counts().to_dict(),
            "avg_word_count": round(full_df["word_count"].mean(), 1),
        },
    }

    joblib.dump(results_payload, os.path.join(MODEL_DIR, "results.pkl"))

    # Save the processed dataset for the dashboard analytics page
    full_df.to_csv(os.path.join(MODEL_DIR, "processed_tickets.csv"), index=False)

    print(f"✓ All files saved to models/ folder")
    print("\nDone! Now we can run streamlit run app.py")


# ══════════════════════════════════════════════════════════════
# RUN THE SCRIPT
# ══════════════════════════════════════════════════════════════

# This block only runs when you execute this file directly
# i.e. python src/train_models.py
if __name__ == "__main__":
    train_and_save()



