"""Figure 1 – framework diagram for the revised pipeline."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "figures", "fig_framework.png")
plt.rcParams.update({"font.family": "DejaVu Serif"})
BLUE, AMBER, RED, GREEN = ("#D6E6FA", "#1F4E9A"), ("#FBE3A0", "#B07D12"), ("#F6B3AE", "#B3261E"), ("#A9DBA0", "#2E7D32")

fig, ax = plt.subplots(figsize=(10, 12.6)); ax.set_xlim(0, 100); ax.set_ylim(-6, 126); ax.axis("off")


def box(y, h, title, sub, col, x=4, w=70, fs=11.5):
    fc, ec = col
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="top", fontsize=fs, fontweight="bold")
    ax.text(x + w / 2, y + h - 5.0, sub, ha="center", va="top", fontsize=8.9, linespacing=1.35)
    return y


def arrow(y1, y2, x=39, text=None):
    ax.annotate("", xy=(x, y2), xytext=(x, y1), arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#222"))
    if text:
        ax.text(x + 1.5, (y1 + y2) / 2, text, va="center", fontsize=8.8, style="italic", color="#333")


def brace(y0, y1, col, label, sub):
    ec = col[1]
    ax.plot([76.5, 78, 78, 76.5], [y1, y1, y0, y0], color=ec, lw=2)
    ax.plot([78, 79.5], [(y0 + y1) / 2] * 2, color=ec, lw=2)
    ax.add_patch(FancyBboxPatch((80.5, (y0 + y1) / 2 - 6), 17.5, 12, boxstyle="round,pad=0.3,rounding_size=1", fc=col[0], ec=ec, lw=1.5))
    ax.text(89.25, (y0 + y1) / 2 + 2.0, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(89.25, (y0 + y1) / 2 - 3.2, sub, ha="center", va="center", fontsize=8.5, linespacing=1.2)


ax.add_patch(FancyBboxPatch((4, 118.5), 94, 7, boxstyle="round,pad=0.25,rounding_size=1.2", fc="#EEF4FC", ec="#1F4E9A", lw=1.6))
ax.text(51, 122, "Irrigation-need classification, label-rule recovery and\nwater–yield-risk Pareto analysis",
        ha="center", va="center", fontsize=12, fontweight="bold", linespacing=1.2)
ys = []
specs = [
    (10, "Public Kaggle dataset", "10,000 records · 19 features · 3-class Irrigation_Need · MIT licence", BLUE),
    (11, "Dataset audit and label-rule recovery", "Marginal distributions · ANOVA / χ² with Holm and BH correction\nShallow trees · exhaustive point-score search → exact six-condition rule", BLUE),
    (11, "Leakage-safe pipeline", "Stratified 70 / 15 / 15 split · scaling + one-hot encoding refitted in every fold\nTuning and model selection on training + validation data only", BLUE),
    (11, "Nine classifiers, equal tuning budget", "LR · DT · RF · SVM · GB · XGBoost · LightGBM · CatBoost · MLP\nOptuna TPE, 30 trials each · class weighting searched", BLUE),
    (12, "Evaluation and explainability", "Single test evaluation · bootstrap CIs · QWK · calibration\n5 × 10 repeated CV · corrected t-test / Wilcoxon + Holm\nSHAP · partial dependence · permutation and drop-column importance", BLUE),
]
y = 117
for h, t, s, c in specs:
    y -= h + 2.2; box(y, h, t, s, c); ys.append((y, h))
for (ya, ha), (yb, hb) in zip(ys[:-1], ys[1:]):
    arrow(ya - 0.3, yb + hb + 0.3)
y_pred_bottom = ys[-1][0]
y -= 15.5; box(y, 13, "Derived layer (assumption-labelled)",
               "FAO-56 water balance: TAW, RAW, D_r, Kₛ (Soil_Type, Crop_Type, stage)\nRisk index R = Kᵧ (1 − Kₛ), FAO-33 stage-specific Kᵧ\nPumping energy E = ρgVH / (ηₚ · 3.6×10⁶), H = static lift + operating head",
               AMBER)
y_der = y
arrow(y_pred_bottom - 0.3, y_der + 13.3, text="predicted class ĉ → class-anchored depth d_c")
y -= 13.5; box(y, 11, "Optimization: water vs yield risk",
               "min V(d) = 10·A·d,  min R(d),  d ∈ [5, 80] mm;  E = κV reported\nExact Pareto set [d_min, d*]: closed form + 4,001-point enumeration", RED)
y_opt = y; arrow(y_der - 0.3, y_opt + 11.3)
y -= 13.5; box(y, 11, "Verification and coupling tests",
               "NSGA-II vs MOPSO-CD vs exact front (10 seeds, 2,400 evaluations)\nClass-anchored depth and search window vs unrestricted search", RED)
y_ver = y; arrow(y_opt - 0.3, y_ver + 11.3)
y -= 12.5; box(y, 10, "Decision and uncertainty",
               "Decision-maker chooses a point on the exact front\nJoint Monte-Carlo propagation of all assumptions (2,000 draws)", GREEN)
y_dec = y; arrow(y_ver - 0.3, y_dec + 10.3)
brace(y_pred_bottom, 117 - 0.5, BLUE, "Prediction\nlayer", "classification of\nirrigation need")
brace(y_der, y_der + 13, AMBER, "Derived\nlayer", "explicit\nassumptions")
brace(y_dec, y_opt + 11, RED, "Optimization /\ndecision", "exact front +\nbenchmark")
plt.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white"); print(OUT)
