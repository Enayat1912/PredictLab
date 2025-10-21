import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report, precision_recall_curve
)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="CSV ML Dashboard", page_icon="📊", layout="wide")
st.title("📊 Upload-a-CSV ML Dashboard — Train, Evaluate, Predict")

st.markdown(
    "Upload a **CSV** where the **last column is your dependent variable** (classification). "
    "Then choose which **feature columns are categorical**. This app handles **missing values** automatically."
)

# -----------------------------
# Helpers
# -----------------------------
def fig_download_button(fig, label, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    st.download_button(label, data=buf.getvalue(), file_name=filename, mime="image/png")

def df_download_button(df, label, filename):
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(label, data=csv_buf.getvalue(), file_name=filename, mime="text/csv")
#
# # -----------------------------
# Sample dataset download (online)
# -----------------------------
with st.expander("Need a sample dataset? Click to download the Banknote Authentication CSV (target last)"):
    st.caption(
        "This pulls the **Banknote Authentication** dataset from the UCI ML repo (via Brownlee’s GitHub), "
        "moves the target column `class` to the end, and offers a one-click download."
    )

    SAMPLE_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/banknote_authentication.csv"
    try:
        # Load dataset
        df_sample = pd.read_csv(
            SAMPLE_URL,
            header=None,
            names=["variance", "skewness", "curtosis", "entropy", "class"]
        )

        # Make sure target is last
        cols = [c for c in df_sample.columns if c != "class"] + ["class"]
        df_sample = df_sample[cols]

        st.write("Preview:", df_sample.head())

        buf = io.StringIO()
        df_sample.to_csv(buf, index=False)
        st.download_button(
            "Download Banknote Authentication (target last)",
            data=buf.getvalue(),
            file_name="banknote_authentication_target_last.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(f"Could not fetch sample dataset online: {e}")





st.divider()

# -----------------------------
# Upload data
# -----------------------------
uploaded = st.file_uploader("Upload your CSV (last column = target)", type=["csv"])

@st.cache_data(show_spinner=False)
def read_csv(file):
    return pd.read_csv(file)

if uploaded is not None:
    df = read_csv(uploaded)
    st.subheader("Dataset preview")
    st.write(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Split X / y (last column is target)
    target_col = df.columns[-1]
    X_raw = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    st.info(f"Using **{target_col}** as the target (last column).")

    # Column typing
    st.subheader("Column types")
    default_cats = [c for c in X_raw.columns if X_raw[c].dtype == "object"]
    categorical_cols = st.multiselect(
        "Select categorical feature columns for One-Hot Encoding",
        options=list(X_raw.columns),
        default=default_cats,
    )
    numeric_cols = [c for c in X_raw.columns if c not in categorical_cols]

    # -----------------------------
    # EDA
    # -----------------------------
    with st.expander("🔎 Quick EDA"):
        c1, c2 = st.columns(2)
        # Class balance
        with c1:
            st.caption("Class balance")
            counts = y.value_counts(dropna=False).rename_axis("class").reset_index(name="count")
            st.dataframe(counts, use_container_width=True)
            fig_bal, ax_bal = plt.subplots(figsize=(4.5, 3.2))
            ax_bal.bar(counts["class"].astype(str), counts["count"])
            ax_bal.set_title("Class balance")
            ax_bal.set_xlabel(target_col); ax_bal.set_ylabel("Count")
            st.pyplot(fig_bal)
            fig_download_button(fig_bal, "Download class balance (PNG)", "class_balance.png")

        # Correlation heatmap (numeric only)
        with c2:
            num_for_corr = X_raw[numeric_cols].select_dtypes(include=[np.number])
            if not num_for_corr.empty:
                st.caption("Correlation heatmap (numeric features)")
                corr = num_for_corr.corr(numeric_only=True)
                fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
                im = ax_corr.imshow(corr, aspect="auto")
                ax_corr.set_title("Correlation heatmap")
                ax_corr.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=90)
                ax_corr.set_yticks(range(len(corr.index)), labels=corr.index)
                fig_corr.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
                st.pyplot(fig_corr)
                fig_download_button(fig_corr, "Download correlation heatmap (PNG)", "correlation_heatmap.png")
            else:
                st.info("No numeric features available for correlation.")

        # Histogram selector
        col_to_plot = st.selectbox("Histogram for numeric column", options=numeric_cols or [])
        if col_to_plot:
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3.2))
            ax_hist.hist(pd.to_numeric(X_raw[col_to_plot], errors="coerce").dropna(), bins=30)
            ax_hist.set_title(f"Histogram — {col_to_plot}")
            st.pyplot(fig_hist)
            fig_download_button(fig_hist, "Download histogram (PNG)", f"hist_{col_to_plot}.png")

    # -----------------------------
    # Sidebar: model & params (model-specific)
    # -----------------------------
    st.sidebar.header("Training controls")
    model_choice = st.sidebar.selectbox(
        "Model",
        [
            "Logistic Regression",
            "Random Forest",
            "SVM (SVC)",
            "KNN",
            "Gradient Boosting",
            "HistGradientBoosting",
        ],
        index=0,
    )
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    params = {}
    if model_choice == "Logistic Regression":
        params["C"] = st.sidebar.slider("C (inverse regularization)", 0.01, 10.0, 1.0, 0.01)
        params["max_iter"] = st.sidebar.slider("Max iterations", 100, 4000, 500, 50)
        params["penalty"] = st.sidebar.selectbox("Penalty", ["l2"])
        params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
    elif model_choice == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 800, 300, 10)
        params["max_depth"] = st.sidebar.slider("max_depth (0 = None)", 0, 40, 0, 1)
        params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, 1)
        params["min_samples_leaf"] = st.sidebar.slider("min_samples_leaf", 1, 10, 1, 1)
    elif model_choice == "SVM (SVC)":
        params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0, 0.01)
        params["gamma"] = st.sidebar.selectbox("gamma", ["scale", "auto"])
    elif model_choice == "KNN":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 50, 5, 1)
        params["weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"])
        params["p"] = st.sidebar.selectbox("Minkowski p", [1, 2], index=1)
    elif model_choice == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 200, 10)
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 5, 3, 1)
    elif model_choice == "HistGradientBoosting":
        params["max_depth"] = st.sidebar.slider("max_depth (None = no limit)", 0, 40, 0, 1)
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 1.0, 0.1, 0.01)

    # -----------------------------
    # Preprocessing: impute + encode + scale
    # -----------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # -----------------------------
    # Build model
    # -----------------------------
    if model_choice == "Logistic Regression":
        clf = LogisticRegression(
            C=params["C"], max_iter=int(params["max_iter"]), penalty=params["penalty"], solver=params["solver"]
        )
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=None if params["max_depth"] == 0 else params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=random_state,
        )
    elif model_choice == "SVM (SVC)":
        # probability=True enables ROC/PR curves
        clf = SVC(kernel=params["kernel"], C=params["C"], gamma=params["gamma"], probability=True, random_state=random_state)
    elif model_choice == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"])
    elif model_choice == "Gradient Boosting":
        clf = GradientBoostingClassifier(
            n_estimators=params["n_estimators"], learning_rate=params["learning_rate"], max_depth=params["max_depth"], random_state=random_state
        )
    elif model_choice == "HistGradientBoosting":
        clf = HistGradientBoostingClassifier(
            max_depth=None if params["max_depth"] == 0 else params["max_depth"], learning_rate=params["learning_rate"], random_state=random_state
        )
    else:
        st.stop()

    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])

    # -----------------------------
    # Train/test split
    # -----------------------------
    strat = y if len(pd.Series(y).unique()) <= 50 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    # -----------------------------
    # Train
    # -----------------------------
    pipe.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Precision (macro)", f"{prec:.3f}")
    c3.metric("Recall (macro)", f"{rec:.3f}")
    c4.metric("F1 (macro)", f"{f1:.3f}")

    # Classification report + download
    report_text = classification_report(y_test, y_pred, zero_division=0)
    with st.expander("Classification report"):
        st.code(report_text, language="text")
        # Also as CSV
        rep_df = pd.DataFrame.from_dict(
            {k: v for k, v in classification_report(y_test, y_pred, output_dict=True, zero_division=0).items() if isinstance(v, dict)},
            orient="index"
        ).reset_index().rename(columns={"index": "class"})
        df_download_button(rep_df, "Download metrics (CSV)", "classification_report.csv")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_title("Confusion Matrix")
    classes = [str(c) for c in np.unique(y)]
    ax_cm.set_xticks(range(len(classes)), labels=classes, rotation=45, ha="right")
    ax_cm.set_yticks(range(len(classes)), labels=classes)
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, v, ha='center', va='center')
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    st.pyplot(fig_cm)
    fig_download_button(fig_cm, "Download confusion matrix (PNG)", "confusion_matrix.png")

    # ROC & PR curves (if probabilities available)
    with st.expander("Curves: ROC & Precision–Recall"):
        try:
            proba = pipe.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=np.unique(y))
            n_classes = y_bin.shape[1]

            # ROC
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, label=f"{classes[i]} (AUC={roc_auc:.2f})")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_title("ROC Curves (OvR)")
            ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR"); ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            fig_download_button(fig_roc, "Download ROC (PNG)", "roc_curves.png")

            # PR
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            for i in range(n_classes):
                precision, recall, _ = precision_recall_curve(y_bin[:, i], proba[:, i])
                pr_auc = auc(recall, precision)
                ax_pr.plot(recall, precision, label=f"{classes[i]} (AUC={pr_auc:.2f})")
            ax_pr.set_title("Precision–Recall (OvR)")
            ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision"); ax_pr.legend(loc="lower left")
            st.pyplot(fig_pr)
            fig_download_button(fig_pr, "Download PR (PNG)", "pr_curves.png")

        except Exception:
            st.info("Curves not available for this setup (model may not support predict_proba).")

    # Feature importance / coefficients
    with st.expander("Feature importance / coefficients"):
        try:
            feature_names = pipe.named_steps["prep"].get_feature_names_out()
            clf_step = pipe.named_steps["clf"]

            if hasattr(clf_step, "feature_importances_"):
                imp = clf_step.feature_importances_
                imp_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
                st.dataframe(imp_df, use_container_width=True)
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                ax_imp.barh(imp_df["feature"], imp_df["importance"])
                ax_imp.invert_yaxis()
                ax_imp.set_title("Feature Importance")
                st.pyplot(fig_imp)
                fig_download_button(fig_imp, "Download feature importance (PNG)", "feature_importance.png")
                df_download_button(imp_df, "Download importances (CSV)", "feature_importances.csv")
            elif hasattr(clf_step, "coef_"):
                coef = clf_step.coef_
                # for binary: shape (1, n_features), for multiclass: (n_classes, n_features)
                abs_mean = np.mean(np.abs(coef), axis=0)
                imp_df = pd.DataFrame({"feature": feature_names, "abs_coef_mean": abs_mean}).sort_values("abs_coef_mean", ascending=False)
                st.dataframe(imp_df, use_container_width=True)
                fig_coef, ax_coef = plt.subplots(figsize=(6, 4))
                ax_coef.barh(imp_df["feature"], imp_df["abs_coef_mean"])
                ax_coef.invert_yaxis()
                ax_coef.set_title("Mean |Coefficient|")
                st.pyplot(fig_coef)
                fig_download_button(fig_coef, "Download coefficients (PNG)", "coefficients.png")
                df_download_button(imp_df, "Download coefficients (CSV)", "coefficients.csv")
            else:
                st.info("Model does not expose importances/coefficients.")
        except Exception as e:
            st.warning(f"Could not compute importances: {e}")

    # -----------------------------
    # Download trained model
    # -----------------------------
    st.subheader("💾 Download trained model")
    model_bytes = pickle.dumps(pipe)
    st.download_button(
        "Download .pkl",
        data=model_bytes,
        file_name=f"trained_model_{model_choice.replace(' ', '').lower()}.pkl",
        mime="application/octet-stream",
    )

    # -----------------------------
    # Single prediction form
    # -----------------------------
    st.subheader("🔮 Single prediction")
    st.caption("Enter values for each feature. For categorical columns, type/select the exact category string.")
    with st.form("single_pred"):
        cols = st.columns(2)
        sample = {}
        for i, col in enumerate(X_raw.columns):
            if col in categorical_cols:
                cat_vals = sorted(map(str, pd.Series(X_raw[col]).dropna().unique().tolist()))[:20]
                if cat_vals:
                    sample[col] = cols[i % 2].selectbox(f"{col} (categorical)", options=["(type custom)"] + cat_vals)
                    if sample[col] == "(type custom)":
                        sample[col] = cols[i % 2].text_input(f"{col} (custom)")
                else:
                    sample[col] = cols[i % 2].text_input(f"{col} (categorical)")
            else:
                default = float(pd.to_numeric(X_raw[col], errors="coerce").dropna().median()) if pd.api.types.is_numeric_dtype(X_raw[col]) else 0.0
                sample[col] = cols[i % 2].number_input(col, value=default)
        submit_single = st.form_submit_button("Predict")

    if submit_single:
        sdf = pd.DataFrame([sample])
        pred = pipe.predict(sdf)[0]
        try:
            prob = pipe.predict_proba(sdf)[0]
            st.success(f"Prediction: **{pred}** — Probabilities: " +
                       ", ".join([f"{cls}: {p:.2f}" for cls, p in zip(np.unique(y), prob)]))
        except Exception:
            st.success(f"Prediction: **{pred}**")

    # -----------------------------
    # Batch predictions on a new CSV (features only)
    # -----------------------------
    st.subheader("📦 Batch predictions on a new CSV (features only)")
    st.caption("Upload a CSV with the same **feature columns** as your training X (exclude the target).")
    batch_file = st.file_uploader("Upload features CSV for scoring", type=["csv"], key="batch")
    if batch_file is not None:
        df_infer = pd.read_csv(batch_file)
        missing = [c for c in X_raw.columns if c not in df_infer.columns]
        if missing:
            st.error(f"Missing required feature columns: {missing}")
        else:
            preds = pipe.predict(df_infer[X_raw.columns])
            out = df_infer.copy()
            out["prediction"] = preds
            try:
                probs = pipe.predict_proba(df_infer[X_raw.columns])
                classes = [str(c) for c in pipe.classes_]
                for i, name in enumerate(classes):
                    out[f"prob_{name}"] = probs[:, i]
            except Exception:
                pass

            st.dataframe(out.head(20), use_container_width=True)
            df_download_button(out, "Download predictions CSV", "predictions.csv")

else:
    st.info("Upload a CSV to get started, or use the **Sample dataset** expander above to download one.")
