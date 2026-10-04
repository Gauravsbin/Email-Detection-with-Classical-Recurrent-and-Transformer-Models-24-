import os
import re
import numpy as np
import pandas as pd
from math import sqrt

def clean_text(t: str) -> str:
    """Clean email text by removing URLs, special characters, and normalizing whitespace."""
    if not isinstance(t, str):
        return ""
    t = re.sub(r"http\S+|www\.\S+", "", t)
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t.lower()).strip()

def load_primary_corpus(csv_path: str) -> pd.DataFrame:
    """Load primary dataset (17,538 clean emails)."""
    df = pd.read_csv(csv_path)
    
    text_col = 'Email Text' if 'Email Text' in df.columns else df.columns[0]
    label_col = 'Email Type' if 'Email Type' in df.columns else df.columns[1]
    
    df = df[[text_col, label_col]].dropna().drop_duplicates().reset_index(drop=True)
    df['text'] = df[text_col].apply(clean_text)
    df['y'] = df[label_col].apply(
        lambda x: 1 if str(x).lower().strip() in ['phishing email', 'phishing', '1'] else 0
    )
    return df[['text', 'y']]

def load_external_corpus(csv_path: str, text_cols=['subject', 'body'], label_val=1) -> pd.DataFrame:
    """Load and format external evaluation corpora."""
    df = pd.read_csv(csv_path)
    
    # Combine subject and body if present
    present_cols = [c for c in text_cols if c in df.columns]
    if present_cols:
        df['text'] = df[present_cols].fillna('').apply(lambda r: ' '.join(r), axis=1)
    else:
        df['text'] = df[df.columns[0]].astype(str)
        
    df['text'] = df['text'].apply(clean_text)
    
    if 'label' in df.columns:
        df['y'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() in ['1', 'phishing', 'spam'] else 0)
    else:
        df['y'] = label_val
        
    return df[df['text'].str.strip() != ''][['text', 'y']].reset_index(drop=True)

def wilson_ci(k: int, n: int, z: float = 1.96):
    """Compute 95% Wilson Score Interval for Binomial Proportions."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def compute_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1, FNR, FPR, and confusion counts."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    lo, hi = wilson_ci(tp + tn, total)
    
    return {
        "acc": acc, "acc_ci_lo": lo, "acc_ci_hi": hi,
        "prec": prec, "rec": rec, "f1": f1,
        "fnr": fnr, "fpr": fpr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def get_fixed_split(df: pd.DataFrame, out_dir: str, test_size: float = 0.2, seed: int = 42):
    """Save or load fixed 80/20 train/test index split (14,030 Dev / 3,508 Test)."""
    os.makedirs(out_dir, exist_ok=True)
    dev_path = os.path.join(out_dir, "dev_indices.npy")
    test_path = os.path.join(out_dir, "test_indices.npy")
    
    if os.path.exists(dev_path) and os.path.exists(test_path):
        dev_idx = np.load(dev_path)
        test_idx = np.load(test_path)
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(df))
        dev_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=df['y'].values, random_state=seed
        )
        np.save(dev_path, dev_idx)
        np.save(test_path, test_idx)
        
    return dev_idx, test_idx
import argparse, os, joblib, numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from common import load_primary_corpus, get_fixed_split, compute_metrics

def get_classical_models(seed=42):
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "SGD": SGDClassifier(random_state=seed),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "MLP": MLPClassifier(max_iter=300, random_state=seed)
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(random_state=seed, eval_metric="logloss")
    except ImportError:
        pass
    return models

def run_classical_experiments(csv_path: str, out_dir: str):
    df = load_primary_corpus(csv_path)
    dev_idx, test_idx = get_fixed_split(df, out_dir)
    
    X_dev, y_dev = df.loc[dev_idx, 'text'].values, df.loc[dev_idx, 'y'].values
    X_test, y_test = df.loc[test_idx, 'text'].values, df.loc[test_idx, 'y'].values
    
    cv_rows, test_rows = [], []
    models = get_classical_models()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)
    
    for name, clf in models.items():
        print(f"Running Classical Model: {name}...")
        
        # 3 x 5-Fold CV on Development Split
        for fold, (tr_i, va_i) in enumerate(rskf.split(X_dev, y_dev)):
            pipe = make_pipeline(TfidfVectorizer(max_features=10000, stop_words='english'), clf)
            pipe.fit(X_dev[tr_i], y_dev[tr_i])
            pred_val = pipe.predict(X_dev[va_i])
            m_val = compute_metrics(y_dev[va_i], pred_val)
            cv_rows.append({"model": name, "fold": fold, **m_val})
            
        # Final Fit on entire Development Split -> Evaluate on Common Test Set
        final_pipe = make_pipeline(TfidfVectorizer(max_features=10000, stop_words='english'), clf)
        final_pipe.fit(X_dev, y_dev)
        pred_test = final_pipe.predict(X_test)
        
        m_test = compute_metrics(y_test, pred_test)
        test_rows.append({"model": name, **m_test})
        
        # Save Predictions and Model
        safe_name = name.replace(" ", "_")
        np.save(os.path.join(out_dir, f"test_pred_{safe_name}.npy"), pred_test)
        joblib.dump(final_pipe, os.path.join(out_dir, "models", f"{safe_name}.joblib"))
        
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "cv_metrics_classical.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, "test_metrics_classical.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_classical_experiments(args.csv, args.out)
import argparse, os, numpy as np, pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from common import load_primary_corpus, get_fixed_split, compute_metrics

RECURRENT_MODELS = {
    "Simple RNN": lambda: layers.SimpleRNN(100),
    "LSTM": lambda: layers.LSTM(100),
    "Bi-LSTM": lambda: layers.Bidirectional(layers.LSTM(100)),
    "GRU": lambda: layers.GRU(100)
}

def build_rnn_model(cell_fn, vocab_size, maxlen=300):
    model = models.Sequential([
        layers.Input(shape=(maxlen,)),
        layers.Embedding(vocab_size, 50, mask_zero=True),
        cell_fn(),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def run_recurrent_experiments(csv_path: str, out_dir: str, maxlen=300, epochs=5):
    df = load_primary_corpus(csv_path)
    dev_idx, test_idx = get_fixed_split(df, out_dir)
    
    X_dev, y_dev = df.loc[dev_idx, 'text'].values, df.loc[dev_idx, 'y'].values
    X_test, y_test = df.loc[test_idx, 'text'].values, df.loc[test_idx, 'y'].values
    
    cv_rows, test_rows = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, cell_fn in RECURRENT_MODELS.items():
        print(f"Running Recurrent Model: {name}...")
        
        # 1 x 5-Fold Cross Validation
        for fold, (tr_i, va_i) in enumerate(skf.split(X_dev, y_dev)):
            tokenizer = Tokenizer(num_words=10000)
            tokenizer.fit_on_texts(X_dev[tr_i])
            
            X_tr_seq = pad_sequences(tokenizer.texts_to_sequences(X_dev[tr_i]), maxlen=maxlen)
            X_va_seq = pad_sequences(tokenizer.texts_to_sequences(X_dev[va_i]), maxlen=maxlen)
            
            model = build_rnn_model(cell_fn, len(tokenizer.word_index) + 1, maxlen)
            model.fit(X_tr_seq, y_dev[tr_i], epochs=epochs, batch_size=64, verbose=0)
            
            pred_val = (model.predict(X_va_seq, batch_size=256, verbose=0).ravel() > 0.5).astype(int)
            cv_rows.append({"model": name, "fold": fold, **compute_metrics(y_dev[va_i], pred_val)})
            
        # Final Model on entire Development Split -> Common Test Set
        final_tok = Tokenizer(num_words=10000)
        final_tok.fit_on_texts(X_dev)
        
        X_dev_seq = pad_sequences(final_tok.texts_to_sequences(X_dev), maxlen=maxlen)
        X_test_seq = pad_sequences(final_tok.texts_to_sequences(X_test), maxlen=maxlen)
        
        final_model = build_rnn_model(cell_fn, len(final_tok.word_index) + 1, maxlen)
        final_model.fit(X_dev_seq, y_dev, epochs=epochs, batch_size=64, verbose=0)
        
        pred_test = (final_model.predict(X_test_seq, batch_size=256, verbose=0).ravel() > 0.5).astype(int)
        test_rows.append({"model": name, **compute_metrics(y_test, pred_test)})
        
        safe_name = name.replace(" ", "_")
        np.save(os.path.join(out_dir, f"test_pred_{safe_name}.npy"), pred_test)
        
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "cv_metrics_recurrent.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, "test_metrics_recurrent.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_recurrent_experiments(args.csv, args.out)
import argparse, os, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
from common import load_primary_corpus, get_fixed_split, compute_metrics

MODEL_NAME = "distilbert-base-cased"

class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item

@torch.no_grad()
def predict_distilbert(model, tokenizer, texts, batch_size=32):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(texts[i:i+batch_size], truncation=True, padding=True, max_length=256, return_tensors="pt").to(model.device)
        logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)

def run_distilbert_experiments(csv_path: str, out_dir: str):
    df = load_primary_corpus(csv_path)
    dev_idx, test_idx = get_fixed_split(df, out_dir)
    
    X_dev, y_dev = df.loc[dev_idx, 'text'].tolist(), df.loc[dev_idx, 'y'].values
    X_test, y_test = df.loc[test_idx, 'text'].tolist(), df.loc[test_idx, 'y'].values
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_rows = []
    test_fold_preds = []
    
    for fold, (tr_i, va_i) in enumerate(skf.split(X_dev, y_dev)):
        print(f"Running DistilBERT - Fold {fold + 1}/5...")
        set_seed(42 + fold)
        
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        
        train_enc = tokenizer([X_dev[i] for i in tr_i], truncation=True, max_length=256)
        train_ds = EmailDataset(train_enc, y_dev[tr_i])
        
        args = TrainingArguments(
            output_dir=f"{out_dir}/distilbert_fold{fold}",
            num_train_epochs=2,
            per_device_train_batch_size=16,
            save_strategy="no",
            logging_steps=100,
            report_to="none"
        )
        
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=DataCollatorWithPadding(tokenizer))
        trainer.train()
        
        # Validate on Fold
        pred_val = predict_distilbert(model, tokenizer, [X_dev[i] for i in va_i])
        cv_rows.append({"model": "DistilBERT", "fold": fold, **compute_metrics(y_dev[va_i], pred_val)})
        
        # Predict on Common Test Set
        pred_test = predict_distilbert(model, tokenizer, X_test)
        test_fold_preds.append(pred_test)
        
    # Aggregate Cross-Fold Test Predictions via Majority Vote
    final_test_pred = (np.mean(test_fold_preds, axis=0) >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, final_test_pred)
    
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, "cv_metrics_distilbert.csv"), index=False)
    pd.DataFrame([{"model": "DistilBERT", **test_metrics}]).to_csv(os.path.join(out_dir, "test_metrics_distilbert.csv"), index=False)
    np.save(os.path.join(out_dir, "test_pred_DistilBERT.npy"), final_test_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_distilbert_experiments(args.csv, args.out)
import argparse, os, joblib, numpy as np, pandas as pd
from common import load_primary_corpus, load_external_corpus, get_fixed_split, compute_metrics

def run_cross_corpus_experiments(primary_csv: str, external_dir: str, out_dir: str):
    df_primary = load_primary_corpus(primary_csv)
    dev_idx, _ = get_fixed_split(df_primary, out_dir)
    X_train, y_train = df_primary.loc[dev_idx, 'text'].values, df_primary.loc[dev_idx, 'y'].values
    
    # Define External Corpora
    ext_files = {
        "Nigerian_5": os.path.join(external_dir, "Nigerian_Fraud.csv"),
        "Enron": os.path.join(external_dir, "Enron.csv"),
        "Nazario": os.path.join(external_dir, "Nazario.csv")
    }
    
    ext_data = {}
    for name, path in ext_files.items():
        if os.path.exists(path):
            ext_data[name] = load_external_corpus(path, label_val=1 if name in ["Nigerian_5", "Nazario"] else 0)
            
    # Add combined Nigerian_5 + Enron test set
    if "Nigerian_5" in ext_data and "Enron" in ext_data:
        ext_data["Nigerian_5 + Enron"] = pd.concat([ext_data["Nigerian_5"], ext_data["Enron"]], ignore_index=True)
        
    results = []
    
    # 1. Classical Representative Models
    for name, model_file in [("Logistic Regression", "Logistic_Regression.joblib"), ("MLP", "MLP.joblib")]:
        model_path = os.path.join(out_dir, "models", model_file)
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            for ext_name, df_ext in ext_data.items():
                preds = model.predict(df_ext['text'].values)
                m = compute_metrics(df_ext['y'].values, preds)
                results.append({"Model": name, "Training Corpus": "Phishing_Email", "External Corpus": ext_name, **m})

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "Table_9_Cross_Corpus.csv"), index=False)
    print("Cross-corpus validation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary_csv", required=True)
    parser.add_argument("--external_dir", required=True)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    run_cross_corpus_experiments(args.primary_csv, args.external_dir, args.out)
import os, subprocess, pandas as pd, numpy as np

def generate_table7(out_dir="results"):
    """Format Table 7: Leakage-Free Repeated Cross-Validation (Mean ± SD)."""
    dfs = []
    for fname in ["cv_metrics_classical.csv", "cv_metrics_recurrent.csv", "cv_metrics_distilbert.csv"]:
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            
    if not dfs:
        return
        
    df_cv = pd.concat(dfs, ignore_index=True)
    summary = df_cv.groupby("model").agg(
        acc_mean=("acc", "mean"), acc_std=("acc", "std"),
        prec_mean=("prec", "mean"), prec_std=("prec", "std"),
        rec_mean=("rec", "mean"), rec_std=("rec", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        fnr_mean=("fnr", "mean"), fnr_std=("fnr", "std"),
        fpr_mean=("fpr", "mean"), fpr_std=("fpr", "std")
    ).reset_index()
    
    t7 = pd.DataFrame({
        "Model": summary["model"],
        "Accuracy": summary.apply(lambda r: f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}", axis=1),
        "Precision": summary.apply(lambda r: f"{r['prec_mean']:.4f} ± {r['prec_std']:.4f}", axis=1),
        "Recall": summary.apply(lambda r: f"{r['rec_mean']:.4f} ± {r['rec_std']:.4f}", axis=1),
        "F1": summary.apply(lambda r: f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}", axis=1),
        "FNR": summary.apply(lambda r: f"{r['fnr_mean']:.4f} ± {r['fnr_std']:.4f}", axis=1),
        "FPR": summary.apply(lambda r: f"{r['fpr_mean']:.4f} ± {r['fpr_std']:.4f}", axis=1)
    })
    
    t7.to_csv("Table_7_Cross_Validation.csv", index=False)
    print("\n=== Table 7: Leakage-free Repeated Cross-Validation Results ===")
    print(t7.to_markdown(index=False))

def generate_table8(out_dir="results"):
    """Format Table 8: Final Common Test-Set Results."""
    labels_path = os.path.join(out_dir, "test_labels.npy")
    if not os.path.exists(labels_path):
        return
    y_test = np.load(labels_path)
    
    models = [
        "Naive Bayes", "Logistic Regression", "SGD", "XGBoost",
        "Decision Tree", "Random Forest", "MLP",
        "Simple RNN", "LSTM", "Bi-LSTM", "GRU", "DistilBERT"
    ]
    
    rows = []
    for model_name in models:
        p_path = os.path.join(out_dir, f"test_pred_{model_name.replace(' ', '_')}.npy")
        if os.path.exists(p_path):
            pred = np.load(p_path)
            from common import compute_metrics
            m = compute_metrics(y_test, pred)
            rows.append({
                "Model": model_name,
                "Accuracy": f"{m['acc']:.4f} [{m['acc_ci_lo']:.4f}, {m['acc_ci_hi']:.4f}]",
                "Precision": f"{m['prec']:.4f}",
                "Recall": f"{m['rec']:.4f}",
                "F1": f"{m['f1']:.4f}",
                "FN": m["fn"],
                "FP": m["fp"]
            })
            
    t8 = pd.DataFrame(rows)
    t8.to_csv("Table_8_Common_Test_Set.csv", index=False)
    print("\n=== Table 8: Final Common Test-Set Results ===")
    print(t8.to_markdown(index=False))

if __name__ == "__main__":
    out_dir = "results"
    primary_csv = "extracted_dataset/phishing_email.csv"
    ext_dir = "extracted_dataset"
    
    # 1. Run Classical
    subprocess.run(["python", "run_classical.py", "--csv", primary_csv, "--out", out_dir])
    
    # 2. Run Recurrent
    subprocess.run(["python", "run_recurrent.py", "--csv", primary_csv, "--out", out_dir])
    
    # 3. Run DistilBERT (GPU required)
    # subprocess.run(["python", "run_distilbert.py", "--csv", primary_csv, "--out", out_dir])
    
    # 4. Run Cross-Corpus
    subprocess.run(["python", "run_cross_corpus.py", "--primary_csv", primary_csv, "--external_dir", ext_dir, "--out", out_dir])
    
    # 5. Format Tables
    generate_table7(out_dir)
    generate_table8(out_dir)

