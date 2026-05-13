import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sentence_transformers import SentenceTransformer
import joblib

# --- Configuration ---
DATA_FILE = "train_clean.csv"          # Your local dataset
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "intent_classifier.pkl")
CONFIDENCE_THRESHOLD = 0.35            # Lower threshold for real-world use

# The six intent labels we support
ALLOWED_LABELS = ["what", "who", "when", "where", "why", "how"]

def load_and_clean(filepath):
    """Load CSV, handle messy lines, normalise labels, and keep only six coarse classes."""
    print(f"Loading dataset from {filepath}...")
    
    # Read with error handling for bad lines
    try:
        df = pd.read_csv(
            filepath,
            on_bad_lines='skip',      # Skip malformed lines
            encoding='utf-8',
            engine='python'           # Python engine tolerates messy quoting
        )
    except Exception as e:
        print(f"  Error reading CSV: {e}")
        print("  Trying with different encoding...")
        df = pd.read_csv(
            filepath,
            on_bad_lines='skip',
            encoding='latin-1',
            engine='python'
        )

    # Standardise column names
    if "Question" in df.columns:
        df.rename(columns={"Question": "question"}, inplace=True)
    if "Type" in df.columns:
        df.rename(columns={"Type": "label"}, inplace=True)

    # Keep only needed columns
    df = df[["question", "label"]]

    # Normalise labels: lowercase, strip whitespace
    df["label"] = df["label"].str.strip().str.lower()

    # Keep only our six allowed labels
    initial_count = len(df)
    df = df[df["label"].isin(ALLOWED_LABELS)]
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with unsupported labels.")

    # Drop missing values
    df.dropna(subset=["question", "label"], inplace=True)

    # Clean questions
    df["question"] = df["question"].astype(str).str.strip()
    df = df[df["question"] != ""]
    df = df[df["question"] != "nan"]

    # Print class distribution
    print(f"\nDataset ready: {len(df)} samples.")
    print("\nClass distribution:")
    for label in ALLOWED_LABELS:
        count = (df["label"] == label).sum()
        print(f"  {label:6s}: {count:4d}")

    return df

def evaluate_model(classifier, X_test, y_test, labels):
    """Compute and display detailed metrics."""
    y_pred = classifier.predict(X_test)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=labels, digits=3))

    print("=" * 60)
    print("  CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    header = "         " + "  ".join(f"{l:>5s}" for l in labels)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {labels[i]:5s}  " + "  ".join(f"{v:5d}" for v in row))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0
    )
    print(f"\n  Macro Precision: {prec:.4f}")
    print(f"  Macro Recall:    {rec:.4f}")
    print(f"  Macro F1-score:  {f1:.4f}   <-- Key metric for imbalanced data")
    print()

def find_best_threshold(classifier, X_val, y_val):
    """Sweep thresholds to find the one that maximises macro F1."""
    proba = classifier.predict_proba(X_val)
    thresholds = np.arange(0.0, 1.0, 0.05)
    best_f1 = 0
    best_t = 0.35

    print("\nSweeping confidence thresholds...")
    for t in thresholds:
        y_pred = []
        for i, probs in enumerate(proba):
            if np.max(probs) < t:
                y_pred.append("unclassified")
            else:
                y_pred.append(classifier.classes_[np.argmax(probs)])
        valid_idx = [j for j, p in enumerate(y_pred) if p != "unclassified"]
        if len(valid_idx) == 0:
            continue
        y_true_filt = [y_val[j] for j in valid_idx]
        y_pred_filt = [y_pred[j] for j in valid_idx]
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_filt, y_pred_filt, labels=ALLOWED_LABELS, average="macro", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"  Best threshold: {best_t:.2f} (F1={best_f1:.4f})")
    return best_t

def train():
    # 1. Load and clean
    df = load_and_clean(DATA_FILE)

    # 2. Encode
    print("\nLoading sentence-transformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("Encoding questions...")
    X = embedder.encode(df["question"].tolist(), show_progress_bar=True)

    y = df["label"].values
    print(f"\nFeature matrix shape: {X.shape}")

    # 3. Split
    print("\nSplitting data (60% train, 20% validation, 20% test)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    print(f"  Train:      {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # 4. Train
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(
        multi_class="multinomial",
        max_iter=2000,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # 5. Cross-validation
    print("\n5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"  CV Macro F1: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean: {cv_scores.mean():.4f}  (+/- {cv_scores.std() * 2:.4f})")

    # 6. Validation evaluation
    print("\nValidation set evaluation:")
    evaluate_model(clf, X_val, y_val, ALLOWED_LABELS)
    best_t = find_best_threshold(clf, X_val, y_val)

    # 7. Test evaluation
    print("\n" + "=" * 60)
    print("  FINAL TEST SET EVALUATION")
    print("=" * 60)
    evaluate_model(clf, X_test, y_test, ALLOWED_LABELS)

    # 8. Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    print(f"\nRecommended confidence threshold: {best_t:.2f}")
    print("(Update CONFIDENCE_THRESHOLD in app.py accordingly)")

if __name__ == "__main__":
    train()