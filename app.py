
import io, os, zipfile, requests
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SmartCrop AI | Paper Implementation", page_icon="🌾", layout="wide")

DATA_FILE = "Crop_recommendation.csv"
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/siddharthss/crop-recommendation-dataset"

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
AC = ["rice","maize","chickpea","blackgram","jute","kidneybeans","lentil","cotton","mothbeans","mungbean","muskmelon"]
HC = ["banana","watermelon","papaya","orange","mango","grapes","coffee","coconut","pomegranate","apple"]

def normalize(s):
    return str(s).strip().lower().replace(" ","").replace("_","")

def standardize_columns(df):
    """Convert common dataset column names into the exact names used by the paper."""
    rename = {}
    for c in df.columns:
        n = normalize(c)
        if n in ("n","nitrogen"):
            rename[c] = "N"
        elif n in ("p","phosphorus"):
            rename[c] = "P"
        elif n in ("k","potassium"):
            rename[c] = "K"
        elif n == "temperature":
            rename[c] = "temperature"
        elif n == "humidity":
            rename[c] = "humidity"
        elif n in ("ph","soilph"):
            rename[c] = "ph"
        elif n == "rainfall":
            rename[c] = "rainfall"
        elif n in ("label","crop"):
            rename[c] = "label"
    return df.rename(columns=rename)

def find_csv_in_zip(z):
    # Prefer the standard crop-recommendation CSV if it exists.
    preferred = [n for n in z.namelist() if os.path.basename(n).lower() == "crop_recommendation.csv"]
    candidates = preferred + [n for n in z.namelist() if n.lower().endswith(".csv") and n not in preferred]
    for name in candidates:
        try:
            with z.open(name) as f:
                test = pd.read_csv(f, nrows=3)
            test = standardize_columns(test)
            if all(c in test.columns for c in FEATURES + ["label"]):
                return name
        except Exception:
            continue
    raise ValueError("No compatible crop-recommendation CSV was found in the downloaded archive.")

@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        df = standardize_columns(df)
    else:
        r = requests.get(DATA_URL, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = find_csv_in_zip(z)
        with z.open(csv_name) as f:
            df = pd.read_csv(f)
        df = standardize_columns(df)
        df.to_csv(DATA_FILE, index=False)

    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset columns detected: {list(df.columns)}\n"
            f"Missing required columns: {missing}"
        )

    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["label"]).copy()
    return df[FEATURES + ["label"]]

def specificity_score(y_true, y_pred, labels):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    vals = []
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        vals.append(tn / (tn + fp) if (tn + fp) else 0)
    return float(np.mean(vals))

@st.cache_resource
def run_paper_methodology(df, category):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.utils import resample
    from xgboost import XGBClassifier

    work = df.copy()
    work["label_norm"] = work["label"].map(normalize)

    if category == "Agricultural":
        keep = AC
    elif category == "Horticultural":
        keep = HC
    else:
        keep = AC + HC

    work = work[work["label_norm"].isin(keep)].copy()
    if work.empty:
        raise ValueError(f"No rows found for {category} crops.")

    # Paper: under-sampling for class imbalance.
    min_count = int(work["label_norm"].value_counts().min())
    sampled = []
    for cls, grp in work.groupby("label_norm"):
        sampled.append(resample(grp, replace=False, n_samples=min_count, random_state=42))
    work = pd.concat(sampled).sample(frac=1, random_state=42).reset_index(drop=True)

    X = work[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(work["label_norm"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_features="sqrt", ccp_alpha=0.001,
            criterion="entropy", random_state=2, max_depth=5
        ),
        "XGBoost": XGBClassifier(
            learning_rate=0.1, max_depth=17, n_estimators=200,
            subsample=0.5, gamma=0, random_state=50,
            objective="multi:softprob", eval_metric="mlogloss",
            n_jobs=4
        ),
        "SVM": SVC(C=5, gamma=0.1, kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            max_depth=6, max_features=5, min_samples_split=4,
            random_state=0, n_estimators=15
        ),
        "KNN": KNeighborsClassifier(n_neighbors=3, metric="minkowski")
    }

    rows, trained = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        try:
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = np.nan
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "Specificity": specificity_score(y_test, pred, list(range(len(le.classes_)))),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "AUC": auc
        })
        trained[name] = model

    metrics = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return trained, metrics, le, work

st.title("🌾 SmartCrop AI")
st.caption("Faithful implementation of the selected 2024 research paper methodology")

with st.expander("📄 Selected Research Paper"):
    st.markdown("""
**Machine learning based recommendation of agricultural and horticultural crop farming in India under the regime of NPK, soil pH and three climatic variables**

**Biplob Dey, Jannatul Ferdous, Romel Ahmed — Heliyon (2024), e25112**

The implementation follows the paper's workflow: crop-condition data, under-sampling, 70:30 train/test split, five ML models, and evaluation using accuracy, precision, recall, specificity, F1 and AUC.
""")

try:
    df = load_data()
except Exception as e:
    st.error("Dataset could not be loaded.")
    st.warning("The app expects Crop_recommendation.csv beside app.py, or it will try to download it automatically.")
    st.exception(e)
    st.stop()

category = st.sidebar.selectbox("Model category", ["Agricultural", "Horticultural", "Mixed"])

models, metrics, le, work = run_paper_methodology(df, category)
best_name = metrics.iloc[0]["Model"]
best_model = models[best_name]

a,b,c,d = st.columns(4)
a.metric("Samples used", f"{len(work):,}")
b.metric("Crop classes", len(le.classes_))
c.metric("Best model", best_name)
d.metric("Best accuracy", f"{metrics.iloc[0]['Accuracy']*100:.2f}%")

st.divider()
st.subheader("🌱 Crop Recommendation")

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", 0.0, 250.0, 60.0)
    P = st.number_input("Phosphorus (P)", 0.0, 250.0, 50.0)
    K = st.number_input("Potassium (K)", 0.0, 250.0, 40.0)
    temp = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
with col2:
    hum = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("🔍 Predict Suitable Crop", type="primary", use_container_width=True):
    row = pd.DataFrame([[N,P,K,temp,hum,ph,rain]], columns=FEATURES)
    pred_num = best_model.predict(row)[0]
    pred = le.inverse_transform([pred_num])[0]
    st.success(f"### Recommended crop: {pred.title()}")

    probs = best_model.predict_proba(row)[0]
    top = np.argsort(probs)[::-1][:3]
    top_df = pd.DataFrame({
        "Crop": [le.inverse_transform([i])[0].title() for i in top],
        "Confidence": [f"{probs[i]*100:.2f}%" for i in top]
    })
    st.dataframe(top_df, hide_index=True, use_container_width=True)

st.divider()
st.subheader("📊 Paper-Methodology Model Comparison")
st.dataframe(metrics.style.format({
    "Accuracy":"{:.2%}", "Precision":"{:.2%}", "Recall":"{:.2%}",
    "Specificity":"{:.2%}", "F1":"{:.2%}", "AUC":"{:.3f}"
}), use_container_width=True, hide_index=True)

st.bar_chart(metrics.set_index("Model")[["Accuracy"]])

st.subheader("🧪 Implementation Workflow")
st.code("""Dataset
  ↓
Agricultural / Horticultural / Mixed classes
  ↓
Under-sampling
  ↓
70% Training + 30% Testing
  ↓
DT | XGBoost | SVM | Random Forest | KNN
  ↓
Accuracy + Precision + Recall + Specificity + F1 + AUC
  ↓
Best model
  ↓
New farm conditions → Crop recommendation""")

st.warning("Academic note: the paper reports a 2,100-record dataset. The accessible public Kaggle copy may differ. Report your local metrics as your implementation results, not as the authors' published results.")
