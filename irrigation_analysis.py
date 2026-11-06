"""
Analysis code for:
"Irrigation-Need Classification, Label-Rule Recovery, and an Assumption-Explicit Water-Yield-Risk Pareto Analysis
for Precision Irrigation: A Computational Demonstration on a Public Dataset"

Reproduces every table and figure. CPU only (Kaggle CPU / Colab / laptop).
Usage:
    python irrigation_analysis.py --stage all        # audit, clf, explain, moo
    python irrigation_analysis.py --stage moo        # one stage
Outputs: outputs/tables/*.csv, outputs/json/*.json, outputs/figures/*.png
"""
import os, sys, json, time, glob, argparse, warnings, platform, itertools
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUT = os.environ.get("OUTDIR") or os.path.join(BASE if not os.path.exists("/kaggle/working") else "/kaggle/working", "outputs")
TD, FD, JD = [os.path.join(OUT, d) for d in ("tables", "figures", "json")]
for d in (TD, FD, JD):
    os.makedirs(d, exist_ok=True)
SEED = 42
ORDER = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
NJOBS = int(os.environ.get("NJOBS", max(1, (os.cpu_count() or 2) - 1)))
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def find_data():
    env = os.environ.get("IRR_DATA")
    if env and os.path.exists(env):
        return env
    cands = glob.glob(os.path.join(BASE, "data", "*.csv")) + glob.glob("/kaggle/input/**/irrigation_prediction.csv", recursive=True)
    if not cands:
        sys.exit("Dataset not found. Set IRR_DATA or put irrigation_prediction.csv in ./data")
    return cands[0]


def load():
    df = pd.read_csv(find_data())
    num = df.select_dtypes("number").columns.tolist()
    cat = [c for c in df.columns if c not in num and c != TARGET]
    return df, num, cat


def save_json(obj, name):
    with open(os.path.join(JD, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def holm(p):
    p = np.asarray(p, float); m = len(p); idx = np.argsort(p); adj = np.empty(m); run = 0
    for r, i in enumerate(idx):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); idx = np.argsort(p)[::-1]; adj = np.empty(m); run = 1.0
    for r, i in enumerate(idx):
        rank = m - r; run = min(run, p[i] * m / rank); adj[i] = run
    return adj


def versions():
    import sklearn
    v = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
         "pandas": pd.__version__, "scipy": __import__("scipy").__version__, "scikit-learn": sklearn.__version__}
    for mod in ("xgboost", "lightgbm", "catboost", "optuna", "shap", "pymoo", "matplotlib"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            v[mod] = "not installed"
    return v


# =====================================================================================
# STAGE 1: DATA AUDIT + RULE RECOVERY
# =====================================================================================
RULE = [("Soil_Moisture < 25 %", lambda d: d.Soil_Moisture < 25, 2),
        ("Rainfall_mm < 300", lambda d: d.Rainfall_mm < 300, 2),
        ("Crop_Growth_Stage in {Flowering, Vegetative}", lambda d: d.Crop_Growth_Stage.isin(["Flowering", "Vegetative"]), 2),
        ("Temperature_C > 30", lambda d: d.Temperature_C > 30, 1),
        ("Wind_Speed_kmh > 10", lambda d: d.Wind_Speed_kmh > 10, 1),
        ("Mulching_Used = No", lambda d: d.Mulching_Used == "No", 1)]


def rule_score(d):
    return sum(w * f(d).astype(int) for _, f, w in RULE)


def rule_predict(d):
    s = rule_score(d)
    return np.where(s >= 7, 2, np.where(s >= 4, 1, 0))


def stage_audit():
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import confusion_matrix
    df, num, cat = load()
    y = df[TARGET].map({c: i for i, c in enumerate(ORDER)}).values
    res = {"n": len(df), "duplicates": int(df.duplicated().sum()), "missing": int(df.isna().sum().sum())}
    # --- marginal distributions: uniformity / equal frequencies
    rows = []
    for c in num:
        x = df[c].values; u = (x - x.min()) / (x.max() - x.min())
        rows.append(dict(feature=c, type="numeric", min=x.min(), max=x.max(), mean=x.mean(), sd=x.std(ddof=1),
                         test="KS vs Uniform(min,max)", p=stats.kstest(u, "uniform").pvalue))
    for c in cat:
        cnt = df[c].value_counts()
        rows.append(dict(feature=c, type=f"categorical ({len(cnt)})", min=cnt.min() / len(df), max=cnt.max() / len(df),
                         test="chi2 vs equal frequencies", p=stats.chisquare(cnt.values).pvalue))
    marg = pd.DataFrame(rows); marg.to_csv(os.path.join(TD, "S1_audit_marginals.csv"), index=False)
    cm = df[num].corr().abs().values; np.fill_diagonal(cm, 0)
    res["max_abs_pearson_between_numeric"] = float(cm.max())
    # Cramer's V among categorical pairs
    cv = []
    for a, b in itertools.combinations(cat, 2):
        ct = pd.crosstab(df[a], df[b]); chi2 = stats.chi2_contingency(ct)[0]
        cv.append(np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))))
    res["max_cramers_v_between_categorical"] = float(max(cv))
    # --- univariate driver tests with Holm and BH correction (Table 6 revised)
    rows = []
    for c in num:
        g = [df.loc[df[TARGET] == k, c] for k in ORDER]
        F, p = stats.f_oneway(*g); H, pk = stats.kruskal(*g)
        eta2 = F * 2 / (F * 2 + len(df) - 3)
        rows.append(dict(feature=c, test="ANOVA F", statistic=F, p=p, effect="eta^2", effect_size=eta2, kruskal_p=pk))
    for c in cat:
        ct = pd.crosstab(df[c], df[TARGET]); chi2, p, dof, _ = stats.chi2_contingency(ct)
        rows.append(dict(feature=c, test="chi2", statistic=chi2, p=p, effect="Cramer's V",
                         effect_size=np.sqrt(chi2 / (len(df) * (min(ct.shape) - 1))), kruskal_p=np.nan))
    t6 = pd.DataFrame(rows); t6["p_holm"] = holm(t6.p); t6["p_bh"] = bh(t6.p)
    t6["significant_holm_0.05"] = t6.p_holm < 0.05
    t6.sort_values("statistic", ascending=False).to_csv(os.path.join(TD, "S2_univariate_tests_corrected.csv"), index=False)
    # --- exact rule recovery
    pred = rule_predict(df)
    res["rule"] = [{"condition": n, "points": w} for n, _, w in RULE]
    res["rule_classes"] = "score 0-3 -> Low; 4-6 -> Medium; 7-9 -> High"
    res["rule_accuracy_all_10000"] = float((pred == y).mean())
    res["rule_confusion"] = confusion_matrix(y, pred).tolist()
    sc = rule_score(df)
    pd.crosstab(sc, df[TARGET]).reindex(columns=ORDER).to_csv(os.path.join(TD, "S3_rule_score_vs_label.csv"))
    # boundary check: are thresholds strict / inclusive?
    res["records_exactly_on_thresholds"] = {"SM==25": int((df.Soil_Moisture == 25).sum()), "Rain==300": int((df.Rainfall_mm == 300).sum()),
                                            "T==30": int((df.Temperature_C == 30).sum()), "Wind==10": int((df.Wind_Speed_kmh == 10).sum())}
    # --- shallow-tree rule recovery from raw features (CV on full data, one-hot)
    X = pd.get_dummies(df.drop(columns=TARGET)).astype(float)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    tr = []
    for depth in [1, 2, 3, 4, 5, 6, 7, 8, 10, None]:
        s = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=SEED), X, y, cv=skf, scoring="accuracy")
        s2 = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=SEED), X, y, cv=skf, scoring="f1_macro")
        tr.append(dict(max_depth=str(depth), cv_accuracy=s.mean(), cv_accuracy_sd=s.std(ddof=1), cv_macro_f1=s2.mean()))
    # tree on the six binarized conditions
    Bn = pd.DataFrame({n: f(df).astype(int) for n, f, _ in RULE})
    for depth in [3, 4, 5, 6]:
        s = cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=SEED), Bn, y, cv=skf, scoring="accuracy")
        tr.append(dict(max_depth=f"{depth} (6 binarized rule conditions)", cv_accuracy=s.mean(), cv_accuracy_sd=s.std(ddof=1),
                       cv_macro_f1=cross_val_score(DecisionTreeClassifier(max_depth=depth, random_state=SEED), Bn, y, cv=skf, scoring="f1_macro").mean()))
    pd.DataFrame(tr).to_csv(os.path.join(TD, "S4_shallow_tree_rule_recovery.csv"), index=False)
    t3 = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X, y)
    res["depth3_tree_text"] = export_text(t3, feature_names=list(X.columns))
    # threshold-learned by a stump on each rule numeric feature (does an unsupervised learner find 25/300/30/10?)
    save_json(res, "S1_audit.json")
    # figure: rule score histogram coloured by label
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ct = pd.crosstab(sc, df[TARGET]).reindex(columns=ORDER).fillna(0)
    bottom = np.zeros(len(ct))
    for k, col in zip(ORDER, ["#4C9F70", "#E0A030", "#C0392B"]):
        ax.bar(ct.index, ct[k], bottom=bottom, color=col, label=k); bottom += ct[k].values
    ax.axvline(3.5, ls="--", c="k", lw=0.8); ax.axvline(6.5, ls="--", c="k", lw=0.8)
    ax.set_xlabel("Additive rule score (0-9)"); ax.set_ylabel("Records"); ax.legend(title="Irrigation_Need", frameon=False)
    ax.set_title("Recovered labelling rule reproduces 10,000/10,000 labels")
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_rule_recovery.png")); plt.close()
    print(json.dumps({k: v for k, v in res.items() if k != "depth3_tree_text"}, indent=1, default=str))


# =====================================================================================
# STAGE 2: CLASSIFICATION
# =====================================================================================
MODELS = ["LogisticRegression", "DecisionTree", "RandomForest", "SVM_RBF", "GradientBoosting",
          "XGBoost", "LightGBM", "CatBoost", "MLP_ANN"]
PRETTY = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest",
          "SVM_RBF": "SVM (RBF)", "GradientBoosting": "Gradient Boosting", "XGBoost": "XGBoost", "LightGBM": "LightGBM",
          "CatBoost": "CatBoost", "MLP_ANN": "MLP (ANN)"}
# library-default configurations = "default"
DEFAULTS = {"LogisticRegression": {}, "DecisionTree": {}, "RandomForest": {"n_estimators": 300}, "SVM_RBF": {},
            "GradientBoosting": {}, "XGBoost": {"n_estimators": 300}, "LightGBM": {"n_estimators": 300},
            "CatBoost": {"iterations": 300}, "MLP_ANN": {}}


def build(name, p, threads=1, proba=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    cw = p.get("cw")
    if name == "LogisticRegression":
        return LogisticRegression(C=p.get("C", 1.0), class_weight=cw, max_iter=5000, random_state=SEED)
    if name == "DecisionTree":
        return DecisionTreeClassifier(max_depth=p.get("max_depth"), min_samples_leaf=p.get("min_samples_leaf", 1),
                                      criterion=p.get("criterion", "gini"), class_weight=cw, random_state=SEED)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=p.get("n_estimators", 100), max_depth=p.get("max_depth"),
                                      min_samples_leaf=p.get("min_samples_leaf", 1), max_features=p.get("max_features", "sqrt"),
                                      class_weight=cw, n_jobs=threads, random_state=SEED)
    if name == "SVM_RBF":
        return SVC(C=p.get("C", 1.0), gamma=p.get("gamma", "scale"), class_weight=cw, probability=proba, random_state=SEED)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=p.get("n_estimators", 100), learning_rate=p.get("learning_rate", 0.1),
                                          max_depth=p.get("max_depth", 3), subsample=p.get("subsample", 1.0), random_state=SEED)
    if name == "XGBoost":
        import xgboost as xgb
        kw = {k: p[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                                "min_child_weight", "reg_lambda") if k in p}
        return xgb.XGBClassifier(**kw, eval_metric="mlogloss", verbosity=0, n_jobs=threads, random_state=SEED)
    if name == "LightGBM":
        import lightgbm as lgb
        kw = {k: p[k] for k in ("n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
                                "colsample_bytree", "reg_lambda") if k in p}
        if "subsample" in kw: kw["subsample_freq"] = 1
        return lgb.LGBMClassifier(**kw, class_weight=cw, verbosity=-1, n_jobs=threads, random_state=SEED)
    if name == "CatBoost":
        import catboost as cb
        kw = {k: p[k] for k in ("iterations", "depth", "learning_rate", "l2_leaf_reg") if k in p}
        return cb.CatBoostClassifier(**kw, auto_class_weights=("Balanced" if cw == "balanced" else None),
                                     verbose=0, thread_count=threads, random_seed=SEED, allow_writing_files=False)
    if name == "MLP_ANN":
        return MLPClassifier(hidden_layer_sizes=p.get("hidden", (64, 32)), alpha=p.get("alpha", 1e-4),
                             learning_rate_init=p.get("lr", 1e-3), max_iter=p.get("max_iter", 500), random_state=SEED)
    raise ValueError(name)


SAMPLE_WEIGHT_MODELS = {"GradientBoosting", "XGBoost"}   # balanced weighting via sample_weight


def make_pipe(name, p, num, cat, threads=1, proba=False):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    return Pipeline([("pre", pre), ("clf", build(name, p, threads, proba))])


def fit_pipe(name, p, X, y, num, cat, threads=1, proba=False):
    from sklearn.utils.class_weight import compute_sample_weight
    pipe = make_pipe(name, p, num, cat, threads, proba)
    if name in SAMPLE_WEIGHT_MODELS and p.get("cw") == "balanced":
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def predict(pipe, X):
    return np.asarray(pipe.predict(X)).ravel().astype(int)


def metr(yt, yp):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, balanced_accuracy_score
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                macro_precision=precision_score(yt, yp, average="macro", zero_division=0),
                macro_recall=recall_score(yt, yp, average="macro"), balanced_accuracy=balanced_accuracy_score(yt, yp),
                qwk=cohen_kappa_score(yt, yp, weights="quadratic"),
                high_recall=recall_score(yt, yp, labels=[2], average="macro"),
                high_precision=precision_score(yt, yp, labels=[2], average="macro", zero_division=0),
                high_f1=f1_score(yt, yp, labels=[2], average="macro"))


def cv_scores(name, p, X, y, splits, num, cat, n_jobs=NJOBS):
    from joblib import Parallel, delayed

    def one(i, tr, te):
        t0 = time.time()
        m = fit_pipe(name, p, X.iloc[tr], y[tr], num, cat)
        d = metr(y[te], predict(m, X.iloc[te])); d["split"] = i; d["fit_s"] = time.time() - t0
        return d
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(one)(i, tr, te) for i, (tr, te) in enumerate(splits)))


def suggest(name, t):
    cw = t.suggest_categorical("cw", [None, "balanced"]) if name != "MLP_ANN" else None
    if name == "LogisticRegression":
        return dict(C=t.suggest_float("C", 1e-3, 1e2, log=True), cw=cw)
    if name == "DecisionTree":
        return dict(max_depth=t.suggest_int("max_depth", 2, 30), min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20),
                    criterion=t.suggest_categorical("criterion", ["gini", "entropy"]), cw=cw)
    if name == "RandomForest":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 400), max_depth=t.suggest_int("max_depth", 4, 40),
                    min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                    max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.5]), cw=cw)
    if name == "SVM_RBF":
        return dict(C=t.suggest_float("C", 0.1, 300, log=True), gamma=t.suggest_float("gamma", 1e-3, 0.3, log=True), cw=cw)
    if name == "GradientBoosting":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 400), learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    max_depth=t.suggest_int("max_depth", 2, 6), subsample=t.suggest_float("subsample", 0.6, 1.0), cw=cw)
    if name == "XGBoost":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500), max_depth=t.suggest_int("max_depth", 2, 10),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0), min_child_weight=t.suggest_int("min_child_weight", 1, 10),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True), cw=cw)
    if name == "LightGBM":
        return dict(n_estimators=t.suggest_int("n_estimators", 100, 500), learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=t.suggest_int("num_leaves", 8, 128), min_child_samples=t.suggest_int("min_child_samples", 5, 60),
                    subsample=t.suggest_float("subsample", 0.6, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10.0, log=True), cw=cw)
    if name == "CatBoost":
        return dict(iterations=t.suggest_int("iterations", 200, 600), depth=t.suggest_int("depth", 3, 8),
                    learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 10), cw=cw)
    if name == "MLP_ANN":
        h = t.suggest_categorical("hidden", ["64-32", "128-64", "128-64-32", "256-128"])
        return dict(hidden=tuple(int(v) for v in h.split("-")), alpha=t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                    lr=t.suggest_float("lr", 1e-4, 1e-2, log=True), max_iter=1000)


def ece_top(proba, y, bins=15):
    conf = proba.max(1); pred = proba.argmax(1); acc = (pred == y).astype(float); e = 0
    edges = np.linspace(0, 1, bins + 1)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any(): e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0, c - h), min(1, c + h))


def corrected_ttest(diff, n_train, n_test, k):
    """Nadeau & Bengio (2003) corrected repeated k-fold t-test; diff = per-split score differences."""
    diff = np.asarray(diff, float); J = len(diff); m = diff.mean(); v = diff.var(ddof=1)
    if v == 0: return m, 0.0, (1.0 if m == 0 else 0.0)
    t = m / np.sqrt((1 / J + n_test / n_train) * v)
    return m, t, 2 * stats.t.sf(abs(t), J - 1)


def stage_clf(n_trials=30):
    import optuna
    from sklearn.model_selection import train_test_split, StratifiedKFold, RepeatedStratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    df, num, cat = load()
    X = df.drop(columns=TARGET); y = df[TARGET].map({c: i for i, c in enumerate(ORDER)}).values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True); y_tv = np.concatenate([y_tr, y_va])
    X_te = X_te.reset_index(drop=True)
    splits5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tv, y_tv))
    log = {"n_trainval": len(y_tv), "n_test": len(y_te), "test_class_counts": np.bincount(y_te).tolist(), "versions": versions()}
    # ---------- (a) default configs, 5-fold CV on train+val only (preprocessing fitted inside each fold)
    fold_rows, cv_rows = [], []
    for name in MODELS:
        s = cv_scores(name, DEFAULTS[name], X_tv, y_tv, splits5, num, cat); s["model"] = name; s["config"] = "default"
        fold_rows.append(s); print("default CV", name, round(s.macro_f1.mean(), 4), flush=True)
    # ---------- (b) equal-budget Optuna/TPE tuning of ALL nine models (5-fold CV macro-F1, train+val only)
    best_params, tune_time = {}, {}
    ckpt = os.path.join(JD, "C_tuning_checkpoint.json")
    done = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    for name in MODELS:
        t0 = time.time()
        if name in done:
            bp = suggest(name, optuna.trial.FixedTrial(done[name]["raw"])); tune_time[name] = done[name]["sec"]
        else:
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
            # 3 trials in parallel (threads), 5 folds each in processes -> ~15 workers on a 12-16 core CPU / 4-core Kaggle
            study.optimize(lambda t: cv_scores(name, suggest(name, t), X_tv, y_tv, splits5, num, cat).macro_f1.mean(),
                           n_trials=n_trials, n_jobs=int(os.environ.get("TRIAL_JOBS", 3)))
            bp = suggest(name, optuna.trial.FixedTrial(study.best_params)); tune_time[name] = time.time() - t0
            done[name] = {"raw": study.best_params, "sec": tune_time[name], "best_cv": study.best_value}
            json.dump(done, open(ckpt, "w"), indent=1, default=str)
        best_params[name] = bp
        s = cv_scores(name, bp, X_tv, y_tv, splits5, num, cat); s["model"] = name; s["config"] = "tuned"
        fold_rows.append(s); print("tuned CV", name, round(s.macro_f1.mean(), 4), bp, round(tune_time[name]), "s", flush=True)
    folds = pd.concat(fold_rows); folds.to_csv(os.path.join(TD, "C1_cv5_fold_scores_default_and_tuned.csv"), index=False)
    cvsum = folds.groupby(["model", "config"]).agg(["mean", "std"])[["accuracy", "macro_f1", "qwk", "balanced_accuracy", "high_recall"]]
    cvsum.columns = ["_".join(c) for c in cvsum.columns]; cvsum = cvsum.reset_index()
    cvsum.to_csv(os.path.join(TD, "C2_cv5_summary_default_vs_tuned.csv"), index=False)
    save_json({"best_params": best_params, "tuning_seconds": tune_time, "n_trials_per_model": n_trials}, "C_best_params.json")
    # ---------- (c) model selection on CV only (test set NOT used)
    tuned_cv = cvsum[cvsum.config == "tuned"].set_index("model")
    selected = tuned_cv.macro_f1_mean.idxmax()
    log["selected_by_cv_macro_f1"] = selected
    log["cv_ranking_tuned"] = tuned_cv.macro_f1_mean.sort_values(ascending=False).round(5).to_dict()
    # ---------- (d) single final evaluation on the held-out test set for all models, with bootstrap CIs, calibration
    rng = np.random.default_rng(SEED); B = 2000
    boot_idx = [rng.integers(0, len(y_te), len(y_te)) for _ in range(B)]
    test_rows, perclass, boots, cal, cms = [], [], [], [], {}
    probas = {}
    for name in MODELS:
        for cfg, p in (("default", DEFAULTS[name]), ("tuned", best_params[name])):
            t0 = time.time(); m = fit_pipe(name, p, X_tv, y_tv, num, cat, threads=NJOBS, proba=True); ft = time.time() - t0
            yp = predict(m, X_te); d = metr(y_te, yp); d.update(model=name, config=cfg, fit_s=ft); test_rows.append(d)
            if cfg == "tuned":
                cms[name] = confusion_matrix(y_te, yp).tolist()
                rep = classification_report(y_te, yp, target_names=ORDER, output_dict=True, digits=6)
                for k in ORDER:
                    r = rep[k]; tp = int(((yp == ORDER.index(k)) & (y_te == ORDER.index(k))).sum())
                    lo, hi = wilson(tp, int(r["support"]))
                    perclass.append(dict(model=name, cls=k, precision=r["precision"], recall=r["recall"], f1=r["f1-score"],
                                         support=int(r["support"]), recall_wilson_lo=lo, recall_wilson_hi=hi))
                bm = np.array([[v for v in metr(y_te[b], yp[b]).values()] for b in boot_idx])
                keys = list(metr(y_te, yp).keys())
                for j, k in enumerate(keys):
                    boots.append(dict(model=name, metric=k, point=d[k], ci_lo=np.nanpercentile(bm[:, j], 2.5), ci_hi=np.nanpercentile(bm[:, j], 97.5)))
                pr = m.predict_proba(X_te); probas[name] = pr
                onehot = np.eye(3)[y_te]
                cal.append(dict(model=name, brier_multiclass=np.mean(((pr - onehot) ** 2).sum(1)), ece_top_label=ece_top(pr, y_te),
                                log_loss=float(-np.mean(np.log(np.clip(pr[np.arange(len(y_te)), y_te], 1e-15, 1))))))
        print("test", name, flush=True)
    # rule baseline on test
    rp = rule_predict(df.iloc[0:0]) if False else None
    te_df = X_te.copy(); rule_te = rule_predict(te_df); d = metr(y_te, rule_te); d.update(model="RecoveredRule", config="none", fit_s=0.0); test_rows.append(d)
    pd.DataFrame(test_rows).to_csv(os.path.join(TD, "C3_test_metrics_all_models.csv"), index=False)
    pd.DataFrame(perclass).to_csv(os.path.join(TD, "C4_test_per_class_all_models.csv"), index=False)
    pd.DataFrame(boots).to_csv(os.path.join(TD, "C5_test_bootstrap_CI.csv"), index=False)
    pd.DataFrame(cal).to_csv(os.path.join(TD, "C6_calibration.csv"), index=False)
    save_json(cms, "C_confusion_matrices_tuned.json")
    # reliability diagram for 4 models
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for name in ["XGBoost", "LightGBM", "RandomForest", "MLP_ANN", "LogisticRegression"]:
        pr = probas[name]; conf = pr.max(1); acc = (pr.argmax(1) == y_te)
        edges = np.linspace(0, 1, 11); xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = (conf > a) & (conf <= b)
            if mm.sum() >= 5: xs.append(conf[mm].mean()); ys.append(acc[mm].mean())
        ax.plot(xs, ys, marker="o", ms=3, label=PRETTY[name])
    ax.set_xlabel("Mean top-label confidence"); ax.set_ylabel("Observed accuracy"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Reliability (held-out test, tuned models)"); plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_calibration.png")); plt.close()
    # ---------- (e) repeated CV (5 x 10-fold) on train+val for statistics, tuned configs
    rsk = list(RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=SEED).split(X_tv, y_tv))
    rep_rows = []
    for name in MODELS:
        s = cv_scores(name, best_params[name], X_tv, y_tv, rsk, num, cat); s["model"] = name; rep_rows.append(s)
        print("repcv", name, round(s.macro_f1.mean(), 4), flush=True)
    rep = pd.concat(rep_rows); rep.to_csv(os.path.join(TD, "C7_repeated_cv_5x10_fold_scores.csv"), index=False)
    W = rep.pivot(index="split", columns="model", values="macro_f1")[MODELS]
    fr = stats.friedmanchisquare(*[W[c].values for c in MODELS])
    ranks = W.rank(axis=1, ascending=False).mean().sort_values()
    best = W.mean().idxmax()
    rows = []
    n_te_fold = len(y_tv) / 10; n_tr_fold = len(y_tv) - n_te_fold
    first10 = W.iloc[:10]
    for c in [m for m in MODELS if m != best]:
        diff = (W[best] - W[c]).values
        mdiff, t, p_corr = corrected_ttest(diff, n_tr_fold, n_te_fold, 10)
        d10 = (first10[best] - first10[c]).values
        try:
            wz = stats.wilcoxon(d10, zero_method="zsplit"); w_stat, w_p = wz.statistic, wz.pvalue
        except ValueError:
            w_stat, w_p = np.nan, 1.0
        nz = d10[d10 != 0]; rbc = (np.sum(np.sign(nz) * stats.rankdata(np.abs(nz))) / np.sum(stats.rankdata(np.abs(nz)))) if len(nz) else 0.0
        rows.append(dict(comparison=f"{best} vs {c}", mean_diff_macro_f1=mdiff, corrected_t=t, p_corrected_t=p_corr,
                         wilcoxon10_W=w_stat, wilcoxon10_p=w_p, rank_biserial=rbc, n_zero_diffs_10fold=int((d10 == 0).sum()),
                         n_ties_50splits=int((diff == 0).sum())))
    st = pd.DataFrame(rows); st["p_corrected_t_holm"] = holm(st.p_corrected_t); st["wilcoxon10_p_holm"] = holm(st.wilcoxon10_p)
    st.to_csv(os.path.join(TD, "C8_pairwise_tests_holm.csv"), index=False)
    log.update(friedman_chi2=fr.statistic, friedman_p=fr.pvalue, friedman_blocks=len(W), mean_ranks=ranks.round(3).to_dict(), best_repcv=best)
    # ---------- (f) stability across 10 different random partitions (tuned params, top models)
    stab = []
    for s_ in range(10):
        a, b, ya, yb = train_test_split(X, y, test_size=0.15, random_state=1000 + s_, stratify=y)
        for name in ["XGBoost", "LightGBM", "CatBoost", "RandomForest", "DecisionTree", "GradientBoosting"]:
            m = fit_pipe(name, best_params[name], a, ya, num, cat, threads=NJOBS); d = metr(yb, predict(m, b)); d.update(model=name, partition_seed=1000 + s_)
            stab.append(d)
        d = metr(yb, rule_predict(b)); d.update(model="RecoveredRule", partition_seed=1000 + s_); stab.append(d)
    stab = pd.DataFrame(stab); stab.to_csv(os.path.join(TD, "C10_partition_stability.csv"), index=False)
    save_json(log, "C_log.json")
    print(json.dumps(log, indent=1, default=str)[:3000])


# =====================================================================================
# STAGE 3: EXPLAINABILITY
# =====================================================================================
def stage_explain():
    import shap
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import f1_score
    df, num, cat = load()
    X = df.drop(columns=TARGET); y = df[TARGET].map({c: i for i, c in enumerate(ORDER)}).values
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True); y_tv = np.concatenate([y_tr, y_va]); X_te = X_te.reset_index(drop=True)
    SEL = json.load(open(os.path.join(JD, "C_log.json")))["selected_by_cv_macro_f1"] if os.path.exists(os.path.join(JD, "C_log.json")) else "XGBoost"
    bp = json.load(open(os.path.join(JD, "C_best_params.json")))["best_params"][SEL]
    m = fit_pipe(SEL, bp, X_tv, y_tv, num, cat, threads=NJOBS)
    pre, clf = m.named_steps["pre"], m.named_steps["clf"]
    fn = list(pre.get_feature_names_out()); Xt = pd.DataFrame(pre.transform(X_te), columns=fn)
    orig_of = {f: (f.split("__", 1)[1] if f.startswith("num__") else next(c for c in cat if f.split("__", 1)[1].startswith(c + "_"))) for f in fn}
    # --- gain importance aggregated to original features
    if SEL == "XGBoost":
        gain = pd.Series(clf.get_booster().get_score(importance_type="gain")); gain.index = [fn[int(i[1:])] if i.startswith("f") and i[1:].isdigit() else i for i in gain.index]
        gain = gain.reindex(fn).fillna(0)
    elif SEL == "CatBoost":
        gain = pd.Series(clf.get_feature_importance(), index=fn)          # PredictionValuesChange (CatBoost's split-gain analogue)
    else:
        gain = pd.Series(getattr(clf, "feature_importances_", np.zeros(len(fn))), index=fn)
    gain = gain.groupby(lambda f: orig_of[f]).sum(); gain = gain / gain.sum()
    # --- SHAP
    # exact TreeSHAP via XGBoost's native implementation (shap.TreeExplainer cannot parse xgboost>=3 multi-class base_score)
    if SEL == "XGBoost":
        import xgboost as xgb
        contrib = clf.get_booster().predict(xgb.DMatrix(Xt.values), pred_contribs=True)   # (n, k, f+1)
    elif SEL == "CatBoost":
        from catboost import Pool
        contrib = clf.get_feature_importance(data=Pool(Xt.values), type="ShapValues")      # exact TreeSHAP, (n, k, f+1)
    else:
        ex = shap.TreeExplainer(clf); v = ex.shap_values(Xt); v = np.stack(v, -1) if isinstance(v, list) else np.asarray(v)
        contrib = np.concatenate([np.transpose(v, (0, 2, 1)), np.zeros((v.shape[0], v.shape[2], 1))], -1)
    sv = np.transpose(contrib[:, :, :-1], (0, 2, 1))                                    # (n, f, k)
    abs_mean = np.abs(sv).mean(axis=(0, 2)); shap_imp = pd.Series(abs_mean, index=fn).groupby(lambda f: orig_of[f]).sum()
    shap_high = pd.Series(np.abs(sv[:, :, 2]).mean(0), index=fn).groupby(lambda f: orig_of[f]).sum()
    # beeswarm for High class (direction of effect)
    plt.figure(); shap.summary_plot(sv[:, :, 2], Xt, max_display=14, show=False, plot_size=(7, 5))
    plt.title(f"Class High ({SEL}, test set)", fontsize=11); plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_shap_beeswarm_high.png")); plt.close()
    plt.figure(); shap.summary_plot(sv[:, :, 1], Xt, max_display=14, show=False, plot_size=(7, 5))
    plt.title("Class Medium", fontsize=11); plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_shap_beeswarm_medium.png")); plt.close()
    # --- partial dependence (expected ordinal need = P(Med)+2P(High)) for the four numeric drivers
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    thr = {"Soil_Moisture": 25, "Rainfall_mm": 300, "Temperature_C": 30, "Wind_Speed_kmh": 10}
    for ax, c in zip(axes, thr):
        grid = np.linspace(df[c].min(), df[c].max(), 60); vals = []
        for g in grid:
            Z = X_te.copy(); Z[c] = g; pr = m.predict_proba(Z); vals.append((pr[:, 1] + 2 * pr[:, 2]).mean())
        ax.plot(grid, vals, c="#2E5E8C"); ax.axvline(thr[c], ls="--", c="#C0392B", lw=0.8)
        ax.set_xlabel(c); ax.set_ylabel("E[need score]" if c == "Soil_Moisture" else "")
    fig.suptitle(f"Partial dependence (tuned {SEL}); dashed = recovered rule threshold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_partial_dependence.png"), bbox_inches="tight"); plt.close()
    # --- permutation importance on the held-out test set (original columns)
    pi = permutation_importance(m, X_te, y_te, scoring="f1_macro", n_repeats=20, random_state=SEED, n_jobs=NJOBS)
    perm = pd.Series(pi.importances_mean, index=X_te.columns); perm_sd = pd.Series(pi.importances_std, index=X_te.columns)
    # --- drop-column importance (individual removal, retrain, CV on train+val + test)
    from joblib import Parallel, delayed
    base_test = f1_score(y_te, predict(m, X_te), average="macro")
    splits5 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tv, y_tv))

    def drop_one(c):
        n2 = [v for v in num if v != c]; c2 = [v for v in cat if v != c]
        cvs = []
        for tr, te in splits5:
            mm = fit_pipe(SEL, bp, X_tv.iloc[tr].drop(columns=c), y_tv[tr], n2, c2)
            cvs.append(f1_score(y_tv[te], predict(mm, X_tv.iloc[te].drop(columns=c)), average="macro"))
        mm = fit_pipe(SEL, bp, X_tv.drop(columns=c), y_tv, n2, c2)
        return c, np.mean(cvs), f1_score(y_te, predict(mm, X_te.drop(columns=c)), average="macro")
    cvs_full = []
    for tr, te in splits5:
        mm = fit_pipe(SEL, bp, X_tv.iloc[tr], y_tv[tr], num, cat); cvs_full.append(f1_score(y_tv[te], predict(mm, X_tv.iloc[te]), average="macro"))
    base_cv = np.mean(cvs_full)
    dc = Parallel(n_jobs=NJOBS)(delayed(drop_one)(c) for c in X.columns)
    drop = pd.DataFrame(dc, columns=["feature", "cv_macro_f1_without", "test_macro_f1_without"]).set_index("feature")
    drop["delta_cv"] = base_cv - drop.cv_macro_f1_without; drop["delta_test"] = base_test - drop.test_macro_f1_without
    # --- grouped configurations (non-nested): rule features only; all-but-rule features; all except mulching
    rule_feats = ["Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh", "Crop_Growth_Stage", "Mulching_Used"]
    groups = {"Six rule features only": rule_feats, "All 19 except the six rule features": [c for c in X.columns if c not in rule_feats],
              "All 19 features": list(X.columns), "All except Mulching_Used": [c for c in X.columns if c != "Mulching_Used"],
              "Weather only (5)": ["Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh"]}
    grows = []
    for g, cols in groups.items():
        n2 = [v for v in num if v in cols]; c2 = [v for v in cat if v in cols]
        mm = fit_pipe(SEL, bp, X_tv[cols], y_tv, n2, c2, threads=NJOBS); yp = predict(mm, X_te[cols])
        d = metr(y_te, yp); d["configuration"] = g; d["n_features"] = len(cols); grows.append(d)
    pd.DataFrame(grows).to_csv(os.path.join(TD, "E3_group_configurations.csv"), index=False)
    # --- univariate ranking (from audit) and agreement across methods
    uni = pd.read_csv(os.path.join(TD, "S2_univariate_tests_corrected.csv")).set_index("feature")["effect_size"]
    imp = pd.DataFrame({"gain_share": gain, "mean_abs_shap_all_classes": shap_imp, "mean_abs_shap_high": shap_high,
                        "permutation_drop_macro_f1": perm, "permutation_sd": perm_sd, "drop_column_delta_cv": drop.delta_cv,
                        "drop_column_delta_test": drop.delta_test, "univariate_effect_size": uni}).sort_values("mean_abs_shap_all_classes", ascending=False)
    imp.to_csv(os.path.join(TD, "E1_feature_importance_all_methods.csv"))
    cols = ["gain_share", "mean_abs_shap_all_classes", "permutation_drop_macro_f1", "drop_column_delta_cv", "univariate_effect_size"]
    tau = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            tau.loc[a, b] = stats.kendalltau(imp[a], imp[b]).statistic
    top6 = {c: imp[c].sort_values(ascending=False).index[:6].tolist() for c in cols}
    tau.to_csv(os.path.join(TD, "E2_kendall_tau_between_methods.csv"))
    save_json({"top6_by_method": top6, "rank_order_by_method": {c: imp[c].sort_values(ascending=False).index.tolist() for c in cols},
               "base_cv_macro_f1": base_cv, "base_test_macro_f1": base_test, "model": SEL}, "E_summary.json")
    # figure: importance comparison
    fig, ax = plt.subplots(figsize=(8, 4.4)); top = imp.index[:10]
    norm = imp.loc[top, cols].clip(lower=0); norm = norm / norm.max()
    w = 0.16
    for i, c in enumerate(cols):
        ax.bar(np.arange(len(top)) + (i - 2) * w, norm[c], w, label=c.replace("_", " "))
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=35, ha="right"); ax.set_ylabel("Importance (scaled to max = 1)")
    ax.legend(frameon=False, fontsize=7, ncol=2); ax.set_title(f"Feature importance by five methods (tuned {SEL})")
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_importance_methods.png")); plt.close()
    print(imp.round(4).to_string()); print(tau.round(2).to_string()); print(pd.DataFrame(grows)[["configuration", "accuracy", "macro_f1"]])


# =====================================================================================
# STAGE 4: REVISED DERIVED LAYER + OPTIMIZATION
# =====================================================================================
# All parameters ASSUMED, sourced from FAO-56 (Allen et al. 1998) Tables 19 & 22, FAO-33/66 Ky,
# and Brouwer et al. (1989) application efficiencies.
SOIL = {"Sandy": (0.12, 0.045), "Loamy": (0.25, 0.12), "Silt": (0.29, 0.15), "Clay": (0.36, 0.22)}  # (theta_FC, theta_WP) m3/m3, FAO-56 Table 19 midpoints
CROP = {"Wheat": (1.25, 0.55), "Maize": (1.35, 0.55), "Cotton": (1.35, 0.65), "Rice": (0.75, 0.20),
        "Sugarcane": (1.60, 0.65), "Potato": (0.50, 0.35)}  # (Zr_max m, p), FAO-56 Table 22 mid-range
STAGE_ROOT = {"Sowing": 0.0, "Vegetative": 0.6, "Flowering": 1.0, "Harvest": 1.0}  # fraction of (Zr_max - Zr_min) developed
ZR_MIN = 0.30
KY = {  # stage-specific yield response factors (FAO-33 / FAO-66, rounded); Sowing=establishment, Harvest=ripening
    "Wheat": {"Sowing": 0.20, "Vegetative": 0.20, "Flowering": 0.60, "Harvest": 0.20},
    "Maize": {"Sowing": 0.40, "Vegetative": 0.40, "Flowering": 1.50, "Harvest": 0.20},
    "Cotton": {"Sowing": 0.20, "Vegetative": 0.20, "Flowering": 0.50, "Harvest": 0.25},
    "Rice": {"Sowing": 1.00, "Vegetative": 1.00, "Flowering": 1.00, "Harvest": 1.00},   # no FAO-33 stage values: seasonal Ky assumed
    "Sugarcane": {"Sowing": 0.75, "Vegetative": 0.75, "Flowering": 0.50, "Harvest": 0.10},
    "Potato": {"Sowing": 0.45, "Vegetative": 0.45, "Flowering": 0.80, "Harvest": 0.20}}
EA = {"Drip": 0.90, "Sprinkler": 0.75, "Canal": 0.60}           # field application efficiency (Brouwer et al. 1989)
H_OP = {"Drip": 10.0, "Sprinkler": 30.0, "Canal": 0.0}          # operating pressure head, m (~1 bar, ~3 bar, surface)
H_LIFT = {"Groundwater": 45.0, "River": 12.0, "Reservoir": 18.0, "Rainwater": 5.0}  # static lift, m (Table 4)
ETA_P = 0.65
DEPTH_BY_CLASS = {0: 20.0, 1: 40.0, 2: 65.0}
D_BOUNDS = (5.0, 80.0)


def scen_params(r, P=None):
    P = P or {}
    fc, wp = SOIL[r["Soil_Type"]]; fc *= P.get("fc_mult", 1); wp *= P.get("wp_mult", 1)
    zmax, p = CROP[r["Crop_Type"]]; zmax *= P.get("zr_mult", 1); p = float(np.clip(p + P.get("p_add", 0), 0.1, 0.8))
    zr = ZR_MIN + STAGE_ROOT[r["Crop_Growth_Stage"]] * max(zmax - ZR_MIN, 0)
    taw = 1000 * (fc - wp) * zr
    sm = r["Soil_Moisture"]
    if P.get("sm_mode", "pct_taw") == "pct_taw":
        avail = sm / 100.0                                    # Soil_Moisture read as % of TAW currently available
    else:
        avail = (sm - 8.0) / (65.0 - 8.0)                     # alternative: min-max rescaling of dataset range
    dr0 = (1 - avail) * taw
    ky = KY[r["Crop_Type"]][r["Crop_Growth_Stage"]] * P.get("ky_mult", 1)
    ea = float(np.clip(EA[r["Irrigation_Type"]] + P.get("ea_add", 0), 0.4, 0.95))
    H = H_LIFT[r["Water_Source"]] * P.get("lift_mult", 1) + H_OP[r["Irrigation_Type"]] * P.get("op_mult", 1)
    eta = P.get("eta", ETA_P)
    return dict(taw=taw, raw=p * taw, p=p, dr0=dr0, ky=ky, ea=ea, H=H, eta=eta, area=r["Field_Area_hectare"], zr=zr)


def objectives(d, s):
    """d: gross depth (mm). Returns V (m3, gross), R (relative yield-loss risk index), E (kWh), Ks, drainage (mm)."""
    d = np.asarray(d, float)
    V = 10.0 * s["area"] * d
    net = s["ea"] * d
    dr1 = np.maximum(0.0, s["dr0"] - net)
    drain = np.maximum(0.0, net - s["dr0"])
    ks = np.where(dr1 <= s["raw"], 1.0, (s["taw"] - dr1) / ((1 - s["p"]) * s["taw"]))
    ks = np.clip(ks, 0, 1)
    R = np.clip(s["ky"] * (1 - ks), 0, 1)
    E = 1000 * 9.81 * V * s["H"] / (s["eta"] * 3.6e6)
    return V, R, E, ks, drain


def d_star(s):
    """Analytic end of the Pareto set: smallest gross depth that brings depletion back to RAW (Ks = 1)."""
    return max(0.0, (s["dr0"] - s["raw"]) / s["ea"])


def exact_front(s, lo, hi, n=4001):
    d = np.linspace(lo, hi, n); V, R, E, _, _ = objectives(d, s)
    F = np.column_stack([V, R]); keep = nondom(F)
    return d[keep], F[keep]


def nondom(F):
    F = np.asarray(F); n = len(F); keep = np.ones(n, bool)
    order = np.lexsort(F.T[::-1]); best_r = np.inf
    for i in order:                     # 2-objective sweep (sorted by f1 then f2)
        if F[i, 1] < best_r - 1e-15: best_r = F[i, 1]
        else: keep[i] = False
    # retain exact duplicates of first occurrence only
    return keep


def select_scenarios(df, n=150):
    pool = df[df.Irrigation_Type != "Rainfed"].copy()
    pool["stratum"] = pool.Water_Source + "|" + pool.Crop_Growth_Stage
    per = n // pool.stratum.nunique(); rest = n - per * pool.stratum.nunique()
    parts = [g.sample(per + (1 if i < rest else 0), random_state=SEED) for i, (_, g) in enumerate(pool.groupby("stratum"))]
    sc = pd.concat(parts).reset_index().rename(columns={"index": "record_id"})
    rep = []
    for c in ["Soil_Moisture", "Field_Area_hectare", "Temperature_C", "Rainfall_mm"]:
        rep.append(dict(variable=c, test="KS sample vs pool", p=stats.ks_2samp(sc[c], pool[c]).pvalue))
    for c in ["Crop_Type", "Soil_Type", "Irrigation_Type", "Irrigation_Need", "Mulching_Used"]:
        obs = sc[c].value_counts().reindex(pool[c].unique()).fillna(0); exp = pool[c].value_counts(normalize=True).reindex(obs.index) * len(sc)
        rep.append(dict(variable=c, test="chi2 sample vs pool", p=stats.chisquare(obs, exp).pvalue))
    return sc, pool, pd.DataFrame(rep)


def run_moea(s, algo_name, seed, lo, hi, n_eval=2400, pop=40):
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    class P(Problem):
        def __init__(self): super().__init__(n_var=1, n_obj=2, xl=np.array([lo]), xu=np.array([hi]))
        def _evaluate(self, X, out, *a, **k):
            V, R, _, _, _ = objectives(X[:, 0], s); out["F"] = np.column_stack([V, R])
    if algo_name == "NSGA-II":
        algo = NSGA2(pop_size=pop, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20))
    else:
        from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
        algo = MOPSO_CD(pop_size=pop, sampling=FloatRandomSampling())
    t0 = time.time(); res = minimize(P(), algo, ("n_eval", n_eval), seed=seed, verbose=False)
    F = np.atleast_2d(res.opt.get("F")) if res.opt is not None else np.atleast_2d(res.F)
    return F, time.time() - t0, int(res.algorithm.evaluator.n_eval)


def indicators(F, Fref, ideal, nadir):
    from pymoo.indicators.hv import HV
    from pymoo.indicators.gd import GD
    from pymoo.indicators.igd import IGD
    from pymoo.indicators.igd_plus import IGDPlus
    rng_ = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    Fn = (F - ideal) / rng_; Rn = (Fref - ideal) / rng_
    hv = HV(ref_point=np.array([1.1, 1.1]))
    return dict(hv=hv(Fn), hv_ratio=hv(Fn) / hv(Rn), gd=GD(Rn)(Fn), igd=IGD(Rn)(Fn), igdp=IGDPlus(Rn)(Fn), n_sol=len(F))


def stage_moo(n_seeds=10, n_mc=2000):
    from joblib import Parallel, delayed
    df, num, cat = load()
    sc, pool, rep = select_scenarios(df, 150)
    sc.to_csv(os.path.join(TD, "M0_scenarios_150.csv"), index=False); rep.to_csv(os.path.join(TD, "M0_scenario_representativeness.csv"), index=False)
    # predicted class for each scenario from the tuned classifier (trained without these records)
    from sklearn.model_selection import train_test_split
    SEL = json.load(open(os.path.join(JD, "C_log.json")))["selected_by_cv_macro_f1"] if os.path.exists(os.path.join(JD, "C_log.json")) else "XGBoost"
    bp = json.load(open(os.path.join(JD, "C_best_params.json")))["best_params"][SEL]
    y = df[TARGET].map({c: i for i, c in enumerate(ORDER)}).values
    trn = ~df.index.isin(sc.record_id)
    clf = fit_pipe(SEL, bp, df.loc[trn].drop(columns=TARGET), y[trn], num, cat, threads=NJOBS)
    sc["pred_class"] = predict(clf, sc[df.drop(columns=TARGET).columns]); sc["true_class"] = sc[TARGET].map({c: i for i, c in enumerate(ORDER)})
    lo, hi = D_BOUNDS
    rows, fronts = [], {}
    for i, r in sc.iterrows():
        s = scen_params(r); ds = d_star(s); dfront, F = exact_front(s, lo, hi)
        dc = DEPTH_BY_CLASS[int(r.pred_class)]
        Vc, Rc, Ec, _, drc = objectives(dc, s); Vs, Rs, Es, _, _ = objectives(min(max(ds, lo), hi), s)
        # classifier-informed window
        wlo, whi = max(lo, 0.5 * dc), min(hi, 1.5 * dc)
        dfw, Fw = exact_front(s, wlo, whi)
        kappa = 1000 * 9.81 * s["H"] / (s["eta"] * 3.6e6)  # kWh per m3 (energy = kappa * V)
        rows.append(dict(scenario=i, record_id=r.record_id, crop=r.Crop_Type, soil=r.Soil_Type, stage=r.Crop_Growth_Stage,
                         irrigation_type=r.Irrigation_Type, water_source=r.Water_Source, area_ha=r.Field_Area_hectare,
                         soil_moisture=r.Soil_Moisture, true_class=r.true_class, pred_class=r.pred_class,
                         TAW_mm=s["taw"], RAW_mm=s["raw"], Dr0_mm=s["dr0"], Ea=s["ea"], Ky=s["ky"], H_total_m=s["H"],
                         kWh_per_m3=kappa, d_star_mm=ds, d_star_within_bounds=bool(lo <= ds <= hi),
                         d_front_min=dfront.min(), d_front_max=dfront.max(),
                         R_at_lower_bound=objectives(lo, s)[1], V_at_dstar_m3=Vs, E_at_dstar_kWh=Es, R_at_dstar=Rs,
                         d_class=dc, V_class=Vc, E_class=Ec, R_class=Rc, drain_class_mm=drc,
                         class_depth_dominated=bool(dc > max(ds, lo) + 1e-9 and dc <= hi),
                         excess_water_class_m3=max(0.0, Vc - Vs) if dc > ds else 0.0,
                         window_lo=wlo, window_hi=whi, window_contains_dstar=bool(wlo <= ds <= whi),
                         window_min_risk=Fw[:, 1].min(), window_front_dmax=dfw.max()))
        fronts[i] = (s, dfront, F)
    S = pd.DataFrame(rows); S.to_csv(os.path.join(TD, "M1_scenario_results_exact.csv"), index=False)
    corr_cls = stats.spearmanr(S.pred_class, S.d_star_mm)
    summ = {"n_scenarios": len(S), "share_dstar_within_[5,80]": S.d_star_within_bounds.mean(),
            "share_dstar_below_5mm(no irrigation needed)": (S.d_star_mm < lo).mean(), "share_dstar_above_80": (S.d_star_mm > hi).mean(),
            "share_class_depth_dominated": S.class_depth_dominated.mean(), "mean_excess_water_class_m3": S.excess_water_class_m3.mean(),
            "share_class_depth_with_residual_risk>0": (S.R_class > 1e-9).mean(), "mean_R_class": S.R_class.mean(),
            "share_window_contains_dstar": S.window_contains_dstar.mean(),
            "spearman_pred_class_vs_dstar": [corr_cls.statistic, corr_cls.pvalue],
            "classifier_test_acc_on_150": float((S.pred_class == S.true_class).mean()),
            "d_star_by_pred_class_median": S.groupby("pred_class").d_star_mm.median().to_dict(),
            "d_star_by_pred_class_IQR": S.groupby("pred_class").d_star_mm.quantile([.25, .75]).unstack().to_dict(),
            "corr_water_energy_within_scenario": "exactly 1 (E = kappa*V, kappa constant per scenario)",
            "kWh_per_m3_range": [S.kWh_per_m3.min(), S.kWh_per_m3.max()]}
    # --- algorithm benchmark: exact front as independent reference, normalized indicators, 10 seeds, equal n_eval
    jobs = [(i, a, sd) for i in range(len(S)) for a in ("NSGA-II", "MOPSO-CD") for sd in range(n_seeds)]

    def job(i, a, sd):
        s, dfront, Fref = fronts[i]
        F, t, ne = run_moea(s, a, SEED + sd, lo, hi)
        ideal, nadir = Fref.min(0), Fref.max(0)
        d = indicators(F, Fref, ideal, nadir); d.update(scenario=i, algorithm=a, seed=SEED + sd, runtime_s=t, n_eval=ne); return d
    t0 = time.time()
    B = pd.DataFrame(Parallel(n_jobs=NJOBS)(delayed(job)(*j) for j in jobs)); B.to_csv(os.path.join(TD, "M2_moea_runs_all_seeds.csv"), index=False)
    summ["benchmark_wallclock_s"] = time.time() - t0
    nondeg = set(S.index[S.d_star_mm > lo + 1e-9]); B["nondegenerate_front"] = B.scenario.isin(nondeg)
    B.to_csv(os.path.join(TD, "M2_moea_runs_all_seeds.csv"), index=False)
    summ["n_nondegenerate_scenarios_for_benchmark"] = len(nondeg)
    B = B[B.nondegenerate_front]
    agg = B.groupby(["algorithm"]).agg(["mean", "std", "median"])[["hv_ratio", "gd", "igd", "igdp", "runtime_s", "n_eval", "n_sol"]]
    agg.columns = ["_".join(c) for c in agg.columns]; agg.to_csv(os.path.join(TD, "M3_moea_summary.csv"))
    per = B.groupby(["scenario", "algorithm"])[["hv_ratio", "gd", "igd", "igdp", "runtime_s"]].median().unstack("algorithm")
    trows = []
    for mtr in ["hv_ratio", "gd", "igd", "igdp", "runtime_s"]:
        a, b = per[(mtr, "NSGA-II")].values, per[(mtr, "MOPSO-CD")].values; dlt = a - b
        w = stats.wilcoxon(a, b, zero_method="zsplit") if np.any(dlt != 0) else None
        nz = dlt[dlt != 0]; rbc = np.sum(np.sign(nz) * stats.rankdata(np.abs(nz))) / np.sum(stats.rankdata(np.abs(nz))) if len(nz) else 0
        trows.append(dict(metric=mtr, nsga2_median=np.median(a), mopso_median=np.median(b), median_diff=np.median(dlt),
                          p_wilcoxon=(w.pvalue if w else 1.0), rank_biserial=rbc, n_scenarios=len(a)))
    tt = pd.DataFrame(trows); tt["p_holm"] = holm(tt.p_wilcoxon); tt.to_csv(os.path.join(TD, "M4_moea_tests_holm.csv"), index=False)
    # --- exact enumeration cost (vs MOEA)
    t0 = time.time()
    for i in range(len(S)): exact_front(fronts[i][0], lo, hi)
    summ["exact_enumeration_s_per_scenario"] = (time.time() - t0) / len(S)
    # --- bounds sensitivity
    brow = []
    for (a, b) in [(5, 80), (0, 60), (0, 100), (5, 120)]:
        inb = ((S.d_star_mm >= a) & (S.d_star_mm <= b)).mean()
        brow.append(dict(bounds=f"[{a},{b}]", share_dstar_inside=inb, share_front_truncated_at_upper=(S.d_star_mm > b).mean(),
                         median_front_length_mm=np.median(np.clip(S.d_star_mm, a, b) - a)))
    pd.DataFrame(brow).to_csv(os.path.join(TD, "M5_bounds_sensitivity.csv"), index=False)
    # --- joint Monte-Carlo uncertainty propagation over ALL assumptions (incl. class-to-depth map, Ky/stage weights)
    rng = np.random.default_rng(SEED)
    maps = [{0: 15, 1: 30, 2: 50}, {0: 20, 1: 40, 2: 65}, {0: 25, 1: 50, 2: 80}]
    draws = []
    for k in range(n_mc):
        P = dict(lift_mult=rng.uniform(0.5, 1.5), op_mult=rng.uniform(0.5, 1.5), eta=rng.uniform(0.45, 0.80),
                 ea_add=rng.uniform(-0.10, 0.10), zr_mult=rng.uniform(0.75, 1.25), p_add=rng.uniform(-0.10, 0.10),
                 fc_mult=rng.uniform(0.85, 1.15), wp_mult=rng.uniform(0.85, 1.15), ky_mult=rng.uniform(0.75, 1.25),
                 sm_mode=("pct_taw" if rng.random() < 0.5 else "minmax"), map_id=int(rng.integers(0, 3)))
        i = int(rng.integers(0, len(S))); r = sc.iloc[i]; s = scen_params(r, P); ds = d_star(s)
        dc = maps[P["map_id"]][int(S.pred_class.iloc[i])]
        Vc, Rc, Ec, _, _ = objectives(dc, s); Vs, Rs, Es, _, _ = objectives(np.clip(ds, lo, hi), s)
        draws.append(dict(**{kk: vv for kk, vv in P.items() if kk != "sm_mode"}, sm_mode_pct=int(P["sm_mode"] == "pct_taw"),
                          scenario=i, d_star=ds, E_dstar=Es, V_dstar=Vs, R_class=Rc, class_dominated=int(dc > max(ds, lo)),
                          kwh_per_m3=1000 * 9.81 * s["H"] / (s["eta"] * 3.6e6)))
    MC = pd.DataFrame(draws); MC.to_csv(os.path.join(TD, "M6_joint_uncertainty_draws.csv"), index=False)
    inputs = ["lift_mult", "op_mult", "eta", "ea_add", "zr_mult", "p_add", "fc_mult", "wp_mult", "ky_mult", "sm_mode_pct", "map_id"]
    # importance of each assumption: squared Spearman rank correlation with outputs, after removing scenario effect (per-scenario ranks)
    srows = []
    for out in ["d_star", "kwh_per_m3", "R_class", "class_dominated"]:
        resid = MC[out] - MC.groupby("scenario")[out].transform("mean")
        for inp in inputs:
            rho = stats.spearmanr(MC[inp], resid).statistic
            srows.append(dict(output=out, assumption=inp, spearman_rho=rho, rho2=rho ** 2))
    SA = pd.DataFrame(srows); SA.to_csv(os.path.join(TD, "M7_joint_sensitivity_indices.csv"), index=False)
    summ["mc_share_class_depth_dominated"] = MC.class_dominated.mean()
    summ["mc_by_map_share_dominated"] = MC.groupby("map_id").class_dominated.mean().to_dict()
    summ["mc_by_map_mean_R_class"] = MC.groupby("map_id").R_class.mean().to_dict()
    base_ds = S.d_star_mm.values
    dev = MC.d_star.values - base_ds[MC.scenario.values]
    summ["mc_dstar_deviation_from_baseline_mm_p5_p50_p95"] = np.percentile(dev, [5, 50, 95]).tolist()
    summ["mc_dstar_abs_deviation_median_mm"] = float(np.median(np.abs(dev)))
    summ["mc_share_dstar_changes_more_than_10mm"] = float((np.abs(dev) > 10).mean())
    summ["mc_dstar_p5_p95_by_baseline_class"] = {int(k): np.percentile(MC.d_star[S.pred_class.values[MC.scenario.values] == k], [5, 50, 95]).tolist() for k in range(3)}
    summ["mc_kwh_per_m3_p5_p95"] = MC.kwh_per_m3.quantile([.05, .95]).tolist()
    save_json(summ, "M_summary.json")
    # --- figures
    i0 = int(S.index[(S.d_star_within_bounds) & (S.stage == "Flowering")][0]); s0, d0, F0 = fronts[i0]
    dgrid = np.linspace(lo, hi, 400); Vg, Rg, Eg, _, _ = objectives(dgrid, s0)
    Fn, _, _ = run_moea(s0, "NSGA-II", SEED, lo, hi); Fm, _, _ = run_moea(s0, "MOPSO-CD", SEED, lo, hi)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(dgrid, Rg, c="#2E5E8C"); ax[0].axvline(d_star(s0), ls="--", c="#C0392B", lw=0.9, label="d* (analytic)")
    ax[0].axvline(DEPTH_BY_CLASS[int(S.pred_class[i0])], ls=":", c="k", lw=0.9, label="class-anchored depth")
    ax[0].set_xlabel("Gross depth d (mm)"); ax[0].set_ylabel("Risk index R"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(Vg, Rg, c="0.75", lw=3, label="all depths in [5,80]"); ax[1].plot(F0[:, 0], F0[:, 1], c="#2E5E8C", lw=1.5, label="exact Pareto front")
    ax[1].scatter(Fn[:, 0], Fn[:, 1], s=10, c="#E0A030", label="NSGA-II", zorder=3); ax[1].scatter(Fm[:, 0], Fm[:, 1], s=10, marker="x", c="#4C9F70", label="MOPSO-CD", zorder=3)
    ax[1].set_xlabel("Gross water V (m³)"); ax[1].set_ylabel("Risk index R"); ax[1].legend(frameon=False, fontsize=8)
    ax[2].plot(Vg, Eg, c="#7D3C98"); ax[2].set_xlabel("Gross water V (m³)"); ax[2].set_ylabel("Pumping energy E (kWh)")
    ax[2].set_title(f"E = {1000*9.81*s0['H']/(s0['eta']*3.6e6):.3f} kWh/m³ × V", fontsize=9)
    r0 = sc.iloc[i0]; fig.suptitle(f"Scenario {i0}: {r0.Crop_Type}, {r0.Crop_Growth_Stage}, {r0.Soil_Type}, {r0.Irrigation_Type}, {r0.Water_Source}", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_exact_front_example.png"), bbox_inches="tight"); plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for k, col in zip(range(3), ["#4C9F70", "#E0A030", "#C0392B"]):
        ss = S[S.pred_class == k]; ax[0].scatter(ss.soil_moisture, ss.d_star_mm, s=14, c=col, label=f"predicted {ORDER[k]}")
    for k, v in DEPTH_BY_CLASS.items(): ax[0].axhline(v, ls=":", c=["#4C9F70", "#E0A030", "#C0392B"][k], lw=0.9)
    ax[0].axhspan(lo, hi, color="0.93", zorder=-1); ax[0].set_xlabel("Soil_Moisture (%)"); ax[0].set_ylabel("d* (mm)"); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("Minimum stress-free depth vs class-anchored depths")
    sa = SA[SA.output == "d_star"].set_index("assumption").rho2.sort_values()
    ax[1].barh(sa.index, sa.values, color="#2E5E8C"); ax[1].set_xlabel("ρ² with d* (within-scenario)"); ax[1].set_title("Joint uncertainty: drivers of d*")
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_classifier_vs_dstar_and_sensitivity.png")); plt.close()
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    for a_, mtr in zip(ax, ["hv_ratio", "gd", "igd", "runtime_s"]):
        data = [per[(mtr, "NSGA-II")].values, per[(mtr, "MOPSO-CD")].values]
        a_.boxplot(data, labels=["NSGA-II", "MOPSO-CD"], showfliers=True); a_.set_title({"hv_ratio": "HV / HV(exact) ↑", "gd": "GD (normalized) ↓", "igd": "IGD (normalized) ↓", "runtime_s": "Runtime (s) ↓"}[mtr])
    fig.suptitle(f"150 scenarios × {n_seeds} seeds, 2,400 evaluations each; reference = exact front", y=1.03)
    plt.tight_layout(); plt.savefig(os.path.join(FD, "fig_moea_benchmark.png"), bbox_inches="tight"); plt.close()
    print(json.dumps(summ, indent=1, default=str)); print(agg.round(5).to_string()); print(tt.round(5).to_string())


def stage_orig_check():
    """Common benchmark protocol (3 unnormalized objectives, 1 seed, union reference set) for comparison with the exact-front protocol."""
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.mopso_cd import MOPSO_CD
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.indicators.hv import HV
    from pymoo.indicators.gd import GD
    from pymoo.indicators.igd import IGD
    df, _, _ = load()
    HEAD = {"Groundwater": 45.0, "River": 12.0, "Reservoir": 18.0, "Rainwater": 5.0}

    def yr(depth, sm, st):
        k = {"Flowering": 1.0, "Vegetative": 0.7, "Sowing": 0.5, "Harvest": 0.3}.get(st, 0.6)
        ratio = np.clip((sm + 0.5 * depth) / 70.0, 0, 1.3); ks = np.where(ratio < 1, ratio, 1.0)
        pen = np.where(ratio > 1.3, (ratio - 1.3) * 0.3, 0.0); return k * (1 - np.clip(ks - pen, 0, 1))

    class Pb(Problem):
        def __init__(self, r):
            self.r = r; super().__init__(n_var=1, n_obj=3, xl=np.array([5.0]), xu=np.array([80.0]))
        def _evaluate(self, X, out, *a, **k):
            d = X[:, 0]; V = d * self.r.Field_Area_hectare * 10; E = 1000 * 9.81 * V * HEAD[self.r.Water_Source] / 0.65 / 3.6e6
            out["F"] = np.column_stack([V, E, yr(d, self.r.Soil_Moisture, self.r.Crop_Growth_Stage)])

    def nd(F):
        keep = np.ones(len(F), bool)
        for i in range(len(F)):
            if keep[i]: keep[np.all(F <= F[i], 1) & np.any(F < F[i], 1)] = False
        return F[keep]
    rows = []
    for _, r in df.sample(n=40, random_state=SEED).iterrows():
        rn = minimize(Pb(r), NSGA2(pop_size=40, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20)), ("n_gen", 60), seed=SEED)
        rm = minimize(Pb(r), MOPSO_CD(pop_size=40, sampling=FloatRandomSampling()), ("n_gen", 60), seed=SEED)
        comb = np.vstack([rn.F, rm.F]); ref_pf = nd(comb); ref = comb.max(0) * 1.1 + 1e-6
        rows.append(dict(hv_n=HV(ref_point=ref)(rn.F), hv_m=HV(ref_point=ref)(rm.F), gd_n=GD(ref_pf)(rn.F), gd_m=GD(ref_pf)(rm.F),
                         igd_n=IGD(ref_pf)(rn.F), igd_m=IGD(ref_pf)(rm.F), n_eval_n=rn.algorithm.evaluator.n_eval, n_eval_m=rm.algorithm.evaluator.n_eval,
                         nsol_n=len(rn.F), nsol_m=len(rm.F)))
    O = pd.DataFrame(rows); O.to_csv(os.path.join(TD, "M8_common_protocol_benchmark.csv"), index=False)
    print(O.describe().T[["mean", "std", "50%"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); ap.add_argument("--trials", type=int, default=30)
    a, _ = ap.parse_known_args()
    st = a.stage.split(",")
    t0 = time.time()
    if "audit" in st or "all" in st: stage_audit()
    if "clf" in st or "all" in st: stage_clf(a.trials)
    if "explain" in st or "all" in st: stage_explain()
    if "moo" in st or "all" in st: stage_moo()
    if "orig" in st or "all" in st: stage_orig_check()
    save_json(versions(), "versions.json")
    print("done in", round(time.time() - t0), "s")
