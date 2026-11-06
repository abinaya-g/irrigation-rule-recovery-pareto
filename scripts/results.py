# -*- coding: utf-8 -*-
"""Load rerun outputs into a flat dict R used by the text templates."""
import json, os
import numpy as np, pandas as pd

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
T = lambda f: pd.read_csv(os.path.join(OUT, "tables", f))
J = lambda f: json.load(open(os.path.join(OUT, "json", f)))
PRETTY = {"LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree", "RandomForest": "Random Forest",
          "SVM_RBF": "SVM (RBF)", "GradientBoosting": "Gradient Boosting", "XGBoost": "XGBoost", "LightGBM": "LightGBM",
          "CatBoost": "CatBoost", "MLP_ANN": "MLP (ANN)", "RecoveredRule": "Recovered rule"}
MODELS = list(PRETTY)[:9]


def f(x, n=4):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100 * x:.{n}f}%"


def pv(p):
    if p == 0:
        return "< 10⁻³⁰⁰"
    if p < 1e-4:
        m, e = f"{p:.2e}".split("e"); return f"{float(m):.2f} × 10{sup(int(e))}"
    return f"{p:.4f}" if p < 0.1 else f"{p:.3f}"


def sup(e):
    s = str(e); tbl = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"); return s.translate(tbl)


def load():
    R = {}
    au = J("S1_audit.json"); R["audit"] = au
    R["maxcorr"] = f(au["max_abs_pearson_between_numeric"], 3); R["maxcv"] = f(au["max_cramers_v_between_categorical"], 3)
    R["marg"] = T("S1_audit_marginals.csv"); R["uni"] = T("S2_univariate_tests_corrected.csv"); R["trees"] = pd.read_csv(os.path.join(OUT, "tables", "S4_shallow_tree_rule_recovery.csv"), dtype={"max_depth": str}, keep_default_na=False)
    R["score_tab"] = pd.read_csv(os.path.join(OUT, "tables", "S3_rule_score_vs_label.csv"), index_col=0)
    # classification
    cv = T("C2_cv5_summary_default_vs_tuned.csv"); R["cv"] = cv
    te = T("C3_test_metrics_all_models.csv"); R["test"] = te
    R["pc"] = T("C4_test_per_class_all_models.csv"); R["boot"] = T("C5_test_bootstrap_CI.csv"); R["cal"] = T("C6_calibration.csv")
    R["st"] = T("C8_pairwise_tests_holm.csv"); R["stab"] = T("C10_partition_stability.csv")
    R["repcv"] = T("C7_repeated_cv_5x10_fold_scores.csv").groupby("model").macro_f1.agg(["mean", "std"])
    R["log"] = J("C_log.json"); R["bp"] = J("C_best_params.json"); R["cm"] = J("C_confusion_matrices_tuned.json")
    sel = R["log"]["selected_by_cv_macro_f1"]; R["sel"] = sel; R["sel_pretty"] = PRETTY[sel]
    b = R["boot"]; tt = te[te.config == "tuned"].set_index("model")

    def ci(m, k, n=4):
        r = b[(b.model == m) & (b.metric == k)].iloc[0]; return f"{f(r.ci_lo, n)}–{f(r.ci_hi, n)}"
    R["ci"] = ci
    R["sel_f1"] = f(tt.loc[sel, "macro_f1"]); R["sel_f1_ci"] = ci(sel, "macro_f1"); R["sel_qwk"] = f(tt.loc[sel, "qwk"])
    R["sel_acc"] = f(tt.loc[sel, "accuracy"]); R["sel_acc_ci"] = ci(sel, "accuracy")
    R["tt"] = tt
    # explain
    R["imp"] = pd.read_csv(os.path.join(OUT, "tables", "E1_feature_importance_all_methods.csv"), index_col=0)
    R["tau"] = pd.read_csv(os.path.join(OUT, "tables", "E2_kendall_tau_between_methods.csv"), index_col=0)
    R["grp"] = T("E3_group_configurations.csv"); R["esum"] = J("E_summary.json")
    # moo
    R["msum"] = J("M_summary.json"); R["S"] = T("M1_scenario_results_exact.csv"); R["magg"] = pd.read_csv(os.path.join(OUT, "tables", "M3_moea_summary.csv"), index_col=0)
    R["mt"] = T("M4_moea_tests_holm.csv"); R["bounds"] = T("M5_bounds_sensitivity.csv"); R["sa"] = T("M7_joint_sensitivity_indices.csv")
    R["rep"] = T("M0_scenario_representativeness.csv"); R["orig11"] = T("M8_common_protocol_benchmark.csv")
    R["dom_pct"] = pct(R["msum"]["share_class_depth_dominated"], 0)
    R["versions"] = J("versions.json") if os.path.exists(os.path.join(OUT, "json", "versions.json")) else {}
    return R
