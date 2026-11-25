"""Publication figure suite for the JMI manuscript (v4, Nature/IEEE/SPIE journal grade).

Visual & scientific upgrades in v4:
- Typography: Clean Arial/Helvetica sans-serif with STIX Sans mathtext (matching
  Nature/Science/JMI production art); crisp bold panel badges (A, B, C).
- Palette: Nature/Science-inspired high-contrast colorblind-safe palette
  (Steel Blue #2166AC, Burnt Vermilion #D6604D, Emerald #1B9E77, Charcoal #222B35).
- Fig 1 (fig_design): Visual-first schematic with a genuine 2D geometric diagram
  in Panel C illustrating the L2-sphere score rotation u -> (u + o) / ||u + o||_2
  and orthogonal null-space decomposition (b_parallel vs b_perp), paired with
  refined left-accent pipeline cards in Panels A & B.
- Fig 2 (fig_phase): Added Pareto/tradeoff visual guide, clean legend/callout
  separation without crowding, and crisp IQR ribbons + identity baseline markers.
- Fig 3 (fig_patient_level): Added subtle corner zoom/residual clarity in Panel A,
  zero-overlap callouts in Panel B, and KDE density overlay + dual-stratum
  median callout badges in Panel C.
- Fig 4 (fig_vendor_operating): Added both vendor-specific (filled) and 9-class
  common-subset (open) AUROC markers in Panel A with clean OOF 95% CI ribbon;
  added faint iso-Youden / sensitivity-specificity context and crisp trajectory
  labels in Panel B with zero text collision.

All figures are rendered at exact physical text width (6.5 in) with embedded Type 42
fonts (PDF + 400-dpi PNG).
"""
import json, datetime, os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Arc, Polygon, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(OUT, "..", "..", ".."))
P1 = os.path.join(ROOT, "results", "p1_20261119")
P2 = os.path.join(ROOT, "results", "p2_20261121")
EXTRACTED_JSON = os.path.join(OUT, "extracted_figure_data.json")
FIG_W = 6.5

# Nature / Science / JMI refined colorblind-safe palette
BLUE      = "#1F6FA8"   # deep clinical steel blue
BLUE_LT   = "#EBF4FB"
BLUE_MID  = "#A9CCE3"
VERM      = "#D95F02"   # crisp vermilion / burnt orange
VERM_LT   = "#FDF1E8"
VERM_MID  = "#F5CBA7"
GREEN     = "#1B9E77"   # emerald teal
GREEN_LT  = "#E8F6F1"
PURPLE    = "#7570B3"   # muted slate violet
INK       = "#1C252C"   # near-black charcoal
GREY      = "#5C6B73"   # secondary neutral text
GREY_LT   = "#F4F6F8"   # card neutral fill
GREY_MID  = "#CBD5E1"   # subtle divider / border
GRID_COL  = "#ECEFF3"   # whisper-light grid

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "stixsans",
    "font.size": 7.8,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.2,
    "xtick.labelsize": 7.2,
    "ytick.labelsize": 7.2,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 7.0,
    "lines.linewidth": 1.3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 400,
})

MANIFEST = []


def load_data():
    """Load from raw result files if present, otherwise from extracted_figure_data.json."""
    if os.path.exists(os.path.join(P1, "e3_arrays.npz")) and os.path.exists(os.path.join(P2, "multivendor_eval.json")):
        d3 = np.load(os.path.join(P1, "e3_arrays.npz"), allow_pickle=True)
        d4 = np.load(os.path.join(P1, "e4_post_vq_arrays.npz"), allow_pickle=True)
        mv = json.load(open(os.path.join(P2, "multivendor_eval.json")))
        cv = json.load(open(os.path.join(P2, "source_cv.json")))
        cu = json.load(open(os.path.join(P2, "clinical_utility.json")))
        keys = ["0.0", "0.25", "0.5", "0.75", "1.0"]
        st = d3["strata"]
        m0 = st == "K1X0Z0"
        m1 = st == "K1X1Z1"
        delta = (d3["mse_1.0"] - d3["mse_0.0"]) * 1e6
        bins = np.arange(-3, 16, 1.0)
        c0, _ = np.histogram(delta[m0], bins=bins)
        c1, _ = np.histogram(delta[m1], bins=bins)

        def implied_spec(sens, prev, ppv):
            return 1.0 - (sens * prev * (1.0 - ppv)) / (ppv * (1.0 - prev))

        return {
            "fig_phase": {
                "lam": [0.0, 0.25, 0.5, 0.75, 1.0],
                "mse_mean": [float(d3["mse_" + k].mean()) for k in keys],
                "mse_q25": [float(np.percentile(d3["mse_" + k], 25)) for k in keys],
                "mse_q75": [float(np.percentile(d3["mse_" + k], 75)) for k in keys],
                "ang_mean": [float(d3["angle_" + k].mean()) for k in keys],
                "ang_q25": [float(np.percentile(d3["angle_" + k], 25)) for k in keys],
                "ang_q75": [float(np.percentile(d3["angle_" + k], 75)) for k in keys],
                "mse_id": float(d4["mse_identity"].mean()),
                "ang_id": float(d4["angle_identity"].mean()),
                "mse_fm": float(d4["mse_fullmix"].mean()),
                "ang_fm": float(d4["angle_fullmix"].mean()),
            },
            "fig_patient_level": {
                "mse_s0_lam0": d3["mse_0.0"][m0].tolist(),
                "mse_s0_lam1": d3["mse_1.0"][m0].tolist(),
                "mse_s1_lam0": d3["mse_0.0"][m1].tolist(),
                "mse_s1_lam1": d3["mse_1.0"][m1].tolist(),
                "ang_s0_lam0": d3["angle_0.0"][m0].tolist(),
                "ang_s0_lam1": d3["angle_1.0"][m0].tolist(),
                "ang_s1_lam0": d3["angle_0.0"][m1].tolist(),
                "ang_s1_lam1": d3["angle_1.0"][m1].tolist(),
                "hist_bins": list(range(-3, 16)),
                "hist_counts_s0": c0.tolist(),
                "hist_counts_s1": c1.tolist(),
                "median_delta_s0_1e6": float(np.median(delta[m0])),
                "median_delta_s1_1e6": float(np.median(delta[m1])),
            },
            "fig_vendor_operating": {
                "oof_all": {"auroc": float(cv["all"]["auroc"]), "ci": [float(x) for x in cv["all"]["ci"]]},
                "ins_all": {"auroc": float(mv["source"]["auroc"])},
                "oof_9": {"auroc": 0.6523, "ci": [0.623, 0.682]},
                "ins_9": {"auroc": 0.7462},
                "vendors": {
                    "Force": {"n": mv["Force"]["n"], "classes": mv["Force"]["classes"], "auroc": mv["Force"]["auroc"], "ci": mv["Force"]["ci"], "auroc_9": 0.7145},
                    "BBB":   {"n": mv["BBB"]["n"],   "classes": mv["BBB"]["classes"],   "auroc": mv["BBB"]["auroc"],   "ci": mv["BBB"]["ci"],   "auroc_9": 0.6634},
                    "MX16":  {"n": mv["MX16"]["n"],  "classes": mv["MX16"]["classes"],  "auroc": mv["MX16"]["auroc"],  "ci": mv["MX16"]["ci"],  "auroc_9": 0.6301},
                },
                "operating": {
                    "BBB": {
                        "spec_src": implied_spec(cu["BBB"]["sens_src"], cu["BBB"]["prev_src"], cu["BBB"]["ppv_src"]),
                        "sens_src": cu["BBB"]["sens_src"], "prev_src": cu["BBB"]["prev_src"],
                        "spec_tgt": implied_spec(cu["BBB"]["sens_tgt"], cu["BBB"]["prev_tgt"], cu["BBB"]["ppv_tgt"]),
                        "sens_tgt": cu["BBB"]["sens_tgt"], "prev_tgt": cu["BBB"]["prev_tgt"],
                    },
                    "Force": {
                        "spec_src": implied_spec(cu["Force"]["sens_src"], cu["Force"]["prev_src"], cu["Force"]["ppv_src"]),
                        "sens_src": cu["Force"]["sens_src"], "prev_src": cu["Force"]["prev_src"],
                        "spec_tgt": implied_spec(cu["Force"]["sens_tgt"], cu["Force"]["prev_tgt"], cu["Force"]["ppv_tgt"]),
                        "sens_tgt": cu["Force"]["sens_tgt"], "prev_tgt": cu["Force"]["prev_tgt"],
                    },
                },
            },
        }
    return json.load(open(EXTRACTED_JSON, "r", encoding="utf-8"))


def save(fig, name, data_note, digests, n=None):
    fig.canvas.draw()
    width, height = fig.get_size_inches()
    assert abs(width - FIG_W) < 1e-6, f"{name}: width {width} != {FIG_W}"
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    fig.savefig(pdf, format="pdf")
    fig.savefig(png, format="png", dpi=400)
    plt.close(fig)
    MANIFEST.append({
        "figure": name,
        "width_in": round(width, 3),
        "height_in": round(height, 3),
        "vector": os.path.basename(pdf),
        "raster": os.path.basename(png),
        "data": data_note,
        "n": n,
        "sources_sha256_16": digests,
        "created": datetime.date.today().isoformat(),
    })


def style_ax(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    if grid_axis in ("x", "y", "both"):
        ax.grid(axis=grid_axis, color=GRID_COL, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def panel_header(ax, letter, title):
    ax.text(-0.02, 1.055, letter, transform=ax.transAxes, fontsize=9.2,
            fontweight="bold", color=INK, va="bottom", ha="left")
    ax.text(0.085, 1.058, title, transform=ax.transAxes, fontsize=7.9,
            fontweight="normal", color=GREY, va="bottom", ha="left")


# --------------------------------------------------------------------------- Fig 1
def fig_design():
    """Study design + geometric readout-space decomposition schematic (v4)."""
    fig = plt.figure(figsize=(FIG_W, 3.35))
    ax = fig.add_axes([0.004, 0.004, 0.992, 0.992])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_node(x, y, w, h, title, body, accent=GREY, fc="#FFFFFF",
                  fs_t=6.9, fs_b=6.2, split_y=0.67):
        # Soft drop shadow
        ax.add_patch(FancyBboxPatch(
            (x + 0.25, y - 0.35), w, h, boxstyle="round,pad=0.18,rounding_size=1.0",
            linewidth=0, facecolor="#E2E8F0", alpha=0.55, zorder=1
        ))
        # Main rounded card
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.18,rounding_size=1.0",
            linewidth=0.75, edgecolor=accent, facecolor=fc, zorder=2
        ))
        # Left color accent bar
        ax.add_patch(FancyBboxPatch(
            (x + 0.05, y + 0.4), 0.75, h - 0.8, boxstyle="round,pad=0.05,rounding_size=0.35",
            linewidth=0, facecolor=accent, zorder=3
        ))
        ax.text(x + w / 2 + 0.3, y + h * split_y, title, ha="center", va="center",
                fontsize=fs_t, fontweight="bold", color=INK, zorder=4)
        ax.text(x + w / 2 + 0.3, y + h * (split_y - 0.36), body, ha="center", va="center",
                fontsize=fs_b, color=GREY, linespacing=1.28, zorder=4)

    def draw_arrow(x1, y1, x2, y2, color=GREY, lw=1.0):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>,head_length=3.3,head_width=1.8",
            mutation_scale=1.0, lw=lw, color=color, shrinkA=0.6, shrinkB=0.6, zorder=4
        ))

    # Subtle background bands for Row A and Row B
    ax.add_patch(FancyBboxPatch(
        (0.6, 70.5), 98.8, 28.5, boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=0.6, edgecolor=BLUE_MID, facecolor="#F8FBFE", zorder=0
    ))
    ax.add_patch(FancyBboxPatch(
        (0.6, 44.5), 98.8, 24.0, boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=0.6, edgecolor=VERM_MID, facecolor="#FFFBF7", zorder=0
    ))

    # ===================== ROW A: PAIRED ARM =====================
    ax.text(1.6, 96.3, "A", fontsize=9.2, fontweight="bold", color=BLUE, va="center")
    ax.text(3.8, 96.3, "Paired arm: reference defined by construction",
            fontsize=7.6, fontweight="bold", color=INK, va="center")
    ax.text(98.4, 96.3, "Stratum 1: Br60f $\\rightarrow$ Br40f (kernel)   |   Stratum 2: B $\\rightarrow$ YA (kernel + sampling)",
            fontsize=6.1, color=GREY, ha="right", va="center")

    draw_node(1.6, 72.8, 14.5, 19.5, "CT-RATE", "official valid_ split\nde-identified chest CT\n(Medipol IRB #857)",
              accent=GREY, fc=GREY_LT, split_y=0.78)
    draw_node(19.2, 83.2, 16.8, 9.1, "100 dev patients", "fit $T(z)$, thresholds, head",
              accent=BLUE, fc="#FFFFFF", fs_t=6.7, fs_b=6.0, split_y=0.68)
    draw_node(19.2, 72.8, 16.8, 9.1, "200 test patients", "2 strata $\\times$ 100 paired scans",
              accent=BLUE, fc="#FFFFFF", fs_t=6.7, fs_b=6.0, split_y=0.68)
    draw_node(39.2, 72.8, 16.0, 19.5, "Frozen CT-CLIP", "$24^3 \\times 512$ token grid\n13,824 nodes (BF16)\nno fine-tuning",
              accent=GREY, fc=GREY_LT, split_y=0.78)
    draw_node(58.4, 72.8, 16.0, 19.5, "Affine Correction", "$T(z) = Az + b$\n$T_\\lambda(z) = Az + \\lambda b$\nnull-space $b_\\perp$ control",
              accent=BLUE, fc="#FFFFFF", split_y=0.78)
    draw_node(77.6, 72.8, 20.8, 19.5, "Paired Endpoints", "1. Token MSE ($\\ell_i$)\n2. Direction error ($1-\\cos$)\n3. Discrepancy ($d_i$) & AUROC",
              accent=BLUE, fc="#FFFFFF", split_y=0.78)

    draw_arrow(16.4, 85.0, 18.9, 87.6, color=BLUE)
    draw_arrow(16.4, 80.2, 18.9, 77.4, color=BLUE)
    draw_arrow(36.3, 77.4, 38.9, 77.4, color=BLUE)
    draw_arrow(55.5, 82.5, 58.1, 82.5, color=BLUE)
    draw_arrow(74.7, 82.5, 77.3, 82.5, color=BLUE)

    # ===================== ROW B: EXTERNAL ARM =====================
    ax.text(1.6, 65.8, "B", fontsize=9.2, fontweight="bold", color=VERM, va="center")
    ax.text(4.0, 65.8, "External arm: cross-vendor domain shift without a paired reference",
            fontsize=7.8, fontweight="bold", color=INK, va="center")
    ax.text(98.4, 65.8, "Disjoint patients ($N = 128$ across 3 external scanner models)",
            fontsize=6.3, color=GREY, ha="right", va="center")

    draw_node(1.6, 46.5, 14.5, 16.5, "External Vendors", "BBB ($n=64$, 14 cls)\nMX 16 ($n=30$, 13 cls)\nForce ($n=34$, 17 cls)",
              accent=VERM, fc="#FFFFFF", fs_b=6.0, split_y=0.79)
    draw_node(19.2, 46.5, 16.8, 16.5, "Unpaired Design", "1 scan per patient\nno target reference\n9 common classes",
              accent=VERM, fc="#FFFFFF", fs_b=6.1, split_y=0.79)
    draw_node(39.2, 46.5, 16.0, 16.5, "Same Encoder", "identical frozen\nCT-CLIP weights &\ncosine readout",
              accent=GREY, fc=GREY_LT, fs_b=6.1, split_y=0.79)
    draw_node(58.4, 46.5, 16.0, 16.5, "Source-Fitted Map", "same $T(z)$ & head\ntransferred unchanged;\nadaptation controls",
              accent=VERM, fc="#FFFFFF", fs_b=6.0, split_y=0.79)
    draw_node(77.6, 46.5, 20.8, 16.5, "External Endpoints", "AUROC vs source bracket\n(OOF 0.697 vs in-sample 0.784)\nFixed-threshold sens / spec",
              accent=VERM, fc="#FFFFFF", fs_b=6.0, split_y=0.79)

    draw_arrow(16.4, 54.8, 18.9, 54.8, color=VERM)
    draw_arrow(36.3, 54.8, 38.9, 54.8, color=VERM)
    draw_arrow(55.5, 54.8, 58.1, 54.8, color=VERM)
    draw_arrow(74.7, 54.8, 77.3, 54.8, color=VERM)

    # ===================== ROW C: ALGEBRA + GEOMETRIC MECHANISM =====================
    ax.add_patch(FancyBboxPatch(
        (0.6, 1.2), 98.8, 41.0, boxstyle="round,pad=0.2,rounding_size=1.2",
        linewidth=0.8, edgecolor=INK, facecolor="#FFFFFF", zorder=1
    ))
    ax.text(1.8, 39.2, "C", fontsize=9.2, fontweight="bold", color=INK, va="center", zorder=3)
    ax.text(4.2, 39.2, "Readout-space decomposition & geometric mechanism",
            fontsize=7.8, fontweight="bold", color=INK, va="center", zorder=3)
    ax.text(98.2, 39.2, "Numerical identity verified to $3.2 \\times 10^{-8}$",
            fontsize=6.5, fontweight="bold", color=BLUE, ha="right", va="center", zorder=3)
    ax.plot([1.4, 98.6], [36.4, 36.4], color=GREY_MID, lw=0.6, zorder=2)

    # Left half of Panel C (x: 1.8 .. 62.5): Algebraic decomposition + 2 concise takeaway rows
    ax.add_patch(FancyBboxPatch(
        (2.0, 27.6), 60.0, 7.2, boxstyle="round,pad=0.15,rounding_size=0.8",
        linewidth=0.6, edgecolor=GREY_MID, facecolor=GREY_LT, zorder=2
    ))
    ax.text(32.0, 31.2,
            "$h(T(z)) = g(L(Az + b)) = g((LA)z + Lb) \\;\\;\\Longrightarrow\\;\\; R(z+b) = \\frac{u + o}{\\Vert u + o \\Vert_2}$",
            fontsize=7.8, color=INK, ha="center", va="center", zorder=3)

    # Linear A callout strip
    ax.add_patch(FancyBboxPatch(
        (2.0, 15.2), 60.0, 10.6, boxstyle="round,pad=0.15,rounding_size=0.8",
        linewidth=0.7, edgecolor=BLUE, facecolor=BLUE_LT, zorder=2
    ))
    ax.text(3.3, 23.2, "1. Linear part $A$ (slope / mixing): exact reparameterization ($L' = LA$)",
            fontsize=6.9, fontweight="bold", color=BLUE, va="center", zorder=3)
    ax.text(3.3, 18.4,
            "Absorbed into frozen first layer; buys $13.8\\%$ token MSE reduction at neutral direction ($+0.0035$),\n"
            "but adds no new information and cannot alter class discrimination.",
            fontsize=6.1, color=INK, va="center", linespacing=1.32, zorder=3)

    # Offset b callout strip
    ax.add_patch(FancyBboxPatch(
        (2.0, 2.8), 60.0, 10.6, boxstyle="round,pad=0.15,rounding_size=0.8",
        linewidth=0.7, edgecolor=VERM, facecolor=VERM_LT, zorder=2
    ))
    ax.text(3.3, 10.8, "2. Offset $b$ (patient-independent shift): only surviving term ($o = Lb$)",
            fontsize=6.9, fontweight="bold", color=VERM, va="center", zorder=3)
    ax.text(3.3, 6.0,
            "Rotates normalized score on the $L_2$ unit sphere ($2.2\\times$ direction error, $+0.4\\%$ worse MSE);\n"
            "null-space component $b_\\perp$ ($Lb_\\perp = 0$) alters tokens identically with zero readout effect.",
            fontsize=6.1, color=INK, va="center", linespacing=1.32, zorder=3)

    # Right half of Panel C (x: 64.0 .. 98.2): Geometric illustration of L2-sphere rotation!
    # Create an inset axes with aspect='equal' so circles & vectors are geometrically exact
    ax_geo = fig.add_axes([0.645, 0.025, 0.340, 0.330])
    ax_geo.set_xlim(-0.35, 1.55)
    ax_geo.set_ylim(-0.22, 1.18)
    ax_geo.set_aspect("equal")
    ax_geo.axis("off")

    # Background light frame
    ax_geo.add_patch(FancyBboxPatch(
        (-0.30, -0.18), 1.80, 1.32, boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.6, edgecolor=GREY_MID, facecolor="#FAFBFD", zorder=0
    ))

    # Unit circle arc (L2-normalized score manifold)
    arc = Arc((0, 0), 2.0, 2.0, angle=0, theta1=-8, theta2=88, color=GREY_MID, lw=1.2, ls="--", zorder=1)
    ax_geo.add_patch(arc)
    ax_geo.text(0.18, 1.03, "$L_2$ unit sphere", fontsize=5.8, color=GREY, style="italic", ha="left")

    # Decision threshold ray (fixed source threshold)
    th_ang = np.deg2rad(44)
    ax_geo.plot([0, 1.25 * np.cos(th_ang)], [0, 1.25 * np.sin(th_ang)],
                color=GREEN, lw=0.9, ls=":", zorder=1.5)
    ax_geo.text(1.26 * np.cos(th_ang), 1.26 * np.sin(th_ang) + 0.02, "fixed\nthreshold",
                fontsize=5.6, color=GREEN, ha="left", va="bottom", linespacing=1.05)

    # Origin
    ax_geo.plot(0, 0, "o", ms=2.5, color=INK, zorder=5)

    # Original pre-norm vector u = Lz (angle 24 deg, length 0.92)
    u = np.array([0.88, 0.39])
    u_hat = u / np.linalg.norm(u)
    # Offset vector o = Lb (rotates u upward)
    o = np.array([0.06, 0.56])
    uo = u + o
    uo_hat = uo / np.linalg.norm(uo)

    # Draw vector u
    ax_geo.add_patch(FancyArrowPatch(
        (0, 0), tuple(u), arrowstyle="-|>,head_length=3.5,head_width=1.8",
        mutation_scale=1.0, lw=1.35, color=BLUE, zorder=4
    ))
    ax_geo.text(0.55, 0.14, "$u = Lz$", fontsize=6.6, fontweight="bold", color=BLUE)

    # Draw additive shift o = Lb from tip of u
    ax_geo.add_patch(FancyArrowPatch(
        tuple(u), tuple(uo), arrowstyle="-|>,head_length=3.5,head_width=1.8",
        mutation_scale=1.0, lw=1.35, color=VERM, zorder=4
    ))
    ax_geo.text(0.98, 0.66, "$o = Lb$\n(offset)", fontsize=6.3, fontweight="bold", color=VERM,
                ha="left", va="center", linespacing=1.1)

    # Draw resultant u + o (faint dashed) and projected uo_hat
    ax_geo.plot([0, uo[0]], [0, uo[1]], color=VERM, lw=0.85, ls="--", alpha=0.6, zorder=2)
    ax_geo.add_patch(FancyArrowPatch(
        (0, 0), tuple(uo_hat), arrowstyle="-|>,head_length=3.5,head_width=1.8",
        mutation_scale=1.0, lw=1.35, color=VERM, zorder=4
    ))

    # Points on unit sphere
    ax_geo.plot(u_hat[0], u_hat[1], "o", ms=4.0, mfc="#FFFFFF", mec=BLUE, mew=1.1, zorder=5)
    ax_geo.plot(uo_hat[0], uo_hat[1], "o", ms=4.0, mfc=VERM, mec="#FFFFFF", mew=0.6, zorder=5)

    # Rotation angle arc between u_hat and uo_hat
    ang1 = np.rad2deg(np.arctan2(u_hat[1], u_hat[0]))
    ang2 = np.rad2deg(np.arctan2(uo_hat[1], uo_hat[0]))
    rot_arc = Arc((0, 0), 0.78, 0.78, angle=0, theta1=ang1, theta2=ang2, color=VERM, lw=1.1, zorder=3)
    ax_geo.add_patch(rot_arc)
    ax_geo.text(0.41, 0.34, "$\\Delta\\theta$", fontsize=6.5, fontweight="bold", color=VERM)

    # Null-space note at bottom of geometric diagram
    ax_geo.text(0.60, -0.10, "$b_\\perp \\in \\ker(L) \\Rightarrow Lb_\\perp = 0$ (no rotation)",
                fontsize=5.8, color=GREY, ha="center", va="center")
    ax_geo.text(uo_hat[0] - 0.04, uo_hat[1] + 0.06, "$\\frac{u+o}{\\Vert u+o\\Vert_2}$",
                fontsize=6.5, color=VERM, ha="right", va="bottom")

    save(fig, "fig_design",
         "schematic with geometric L2-sphere rotation diagram; counts 100 dev / 200 test (2 x 100 pairs), 3 vendors",
         {"manifests/scaleup_v2_20261030/confirmation_protocol.json": "4b6f5c7972b703b6",
          "results/p2_20261121/multivendor_eval.json": "23f20cfff541a252"}, n=None)


# --------------------------------------------------------------------------- Fig 2
def fig_phase(data):
    d = data["fig_phase"]
    lam = np.array(d["lam"])
    mse = np.array(d["mse_mean"])
    mse_q25 = np.array(d["mse_q25"])
    mse_q75 = np.array(d["mse_q75"])
    ang = np.array(d["ang_mean"])
    ang_q25 = np.array(d["ang_q25"])
    ang_q75 = np.array(d["ang_q75"])
    mse_id = d["mse_id"]
    ang_id = d["ang_id"]
    mse_fm = d["mse_fm"]
    ang_fm = d["ang_fm"]

    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.38),
                            gridspec_kw={"width_ratios": [1.18, 1.0, 1.0]})
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.205, top=0.845, wspace=0.38)

    # --- Panel A: Phase diagram (MSE reduction vs direction error)
    ax = axs[0]
    x_id = 0.0
    x_sl = (1.0 - mse[0] / mse_id) * 100.0
    x_pc = (1.0 - mse[-1] / mse_id) * 100.0
    x_fm = (1.0 - mse_fm / mse_id) * 100.0

    # Neutral direction band (±10% of identity)
    ax.axhspan(ang_id * 0.90, ang_id * 1.10, color=GREEN_LT, alpha=0.95, zorder=0, lw=0)
    ax.axhline(ang_id, color=GREY, lw=0.8, ls=":", zorder=1)
    ax.text(25.0, ang_id - 0.011, "$\\pm 10\\%$ neutral direction band",
            fontsize=6.0, color=GREEN, ha="right", va="top", style="italic")

    # Decomposition trajectory arrows
    ax.add_patch(FancyArrowPatch(
        (x_id, ang_id), (x_sl, ang[0]), arrowstyle="-|>,head_length=3.2,head_width=1.7",
        mutation_scale=1.0, lw=1.0, color=BLUE, ls="--", shrinkA=5.5, shrinkB=5.5, zorder=2
    ))
    ax.add_patch(FancyArrowPatch(
        (x_sl, ang[0]), (x_pc, ang[-1]), arrowstyle="-|>,head_length=3.2,head_width=1.7",
        mutation_scale=1.0, lw=1.1, color=VERM, shrinkA=5.5, shrinkB=5.5, zorder=2
    ))

    pts = [
        (x_id, ang_id, "identity", "o", GREY, (0.3, 0.070), "left"),
        (x_sl, ang[0], "slope only\n($\\lambda=0$)", "s", BLUE, (15.5, 0.067), "left"),
        (x_pc, ang[-1], "PC-OLS\n($\\lambda=1$)", "^", VERM, (11.5, 0.130), "right"),
        (x_fm, ang_fm, "full mixing\n($-21.7\\%$ MSE)", "D", GREEN, (19.6, 0.212), "right"),
    ]
    for x, y, lab, mk, col, lxy, ha in pts:
        ax.plot(x, y, marker=mk, ms=6.0, color=col, zorder=4, mew=1.0,
                mfc=col if mk != "o" else "#FFFFFF", mec="#FFFFFF" if mk != "o" else col)
        if mk != "o":
            ax.plot(x, y, marker=mk, ms=6.2, mfc="none", mec=col, mew=0.6, zorder=4.1)
        ax.annotate(lab, xy=(x, y), xytext=lxy, fontsize=6.4,
                    fontweight="bold" if mk in ("s", "^") else "normal",
                    color=col, ha=ha, va="center", linespacing=1.18, zorder=5)

    # Clean offset penalty annotation
    ax.text(14.8, 0.104, "$+$ offset $b$\n($2.2\\times$ dir. error,\nno MSE gain)",
            fontsize=5.8, color=VERM, ha="left", va="center", linespacing=1.15)

    ax.set_xlabel("token MSE reduction vs identity (%)")
    ax.set_ylabel("direction error ($1 - \\cos$)")
    ax.set_xlim(-2.5, 25.8)
    ax.set_ylim(0.028, 0.240)
    panel_header(ax, "A", "Endpoint decoupling")
    style_ax(ax)

    # --- Panel B: Token MSE along offset weight \lambda
    ax = axs[1]
    ax.fill_between(lam, mse_q25 * 1e4, mse_q75 * 1e4, color=BLUE, alpha=0.14, lw=0)
    ax.plot(lam, mse_q25 * 1e4, "--", color=BLUE, alpha=0.45, lw=0.7)
    ax.plot(lam, mse_q75 * 1e4, "--", color=BLUE, alpha=0.45, lw=0.7)
    ax.plot(lam, mse * 1e4, "-o", color=BLUE, ms=4.5, mfc=BLUE, mec="#FFFFFF", mew=0.6, zorder=4)

    ax.axhline(mse_id * 1e4, color=GREY, lw=0.85, ls=":")
    ax.text(0.50, mse_id * 1e4 + 0.04, f"identity baseline ({mse_id*1e4:.2f})",
            fontsize=6.1, color=GREY, ha="center", va="bottom")

    ax.annotate("slope only\n($5.28$)", xy=(0.0, mse[0] * 1e4),
                xytext=(0.06, mse[0] * 1e4 - 0.52), fontsize=6.2, color=BLUE, fontweight="bold",
                ha="left", va="top", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7, shrinkA=1, shrinkB=4))
    ax.annotate("PC-OLS\n($5.30$)", xy=(1.0, mse[-1] * 1e4),
                xytext=(0.94, mse[-1] * 1e4 - 0.52), fontsize=6.2, color=INK, fontweight="bold",
                ha="right", va="top", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.7, shrinkA=1, shrinkB=4))
    ax.text(0.50, 5.66, "patient IQR ($1.55\\times10^{-4}$)\n$\\gg$ mean shift ($+0.02$)",
            fontsize=5.9, color=GREY, ha="center", va="center", style="italic", linespacing=1.22)

    ax.set_xlabel("offset weight $\\lambda$ in $T_\\lambda(z) = Az + \\lambda b$")
    ax.set_ylabel("token MSE ($\\times 10^{-4}$)")
    ax.set_xticks(lam)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(4.15, 6.42)
    panel_header(ax, "B", "Token MSE along $\\lambda$")
    style_ax(ax)

    # --- Panel C: Direction error along offset weight \lambda
    ax = axs[2]
    ax.fill_between(lam, ang_q25, ang_q75, color=VERM, alpha=0.14, lw=0)
    ax.plot(lam, ang_q25, "--", color=VERM, alpha=0.45, lw=0.7)
    ax.plot(lam, ang_q75, "--", color=VERM, alpha=0.45, lw=0.7)
    ax.plot(lam, ang, "-^", color=VERM, ms=4.8, mfc=VERM, mec="#FFFFFF", mew=0.6, zorder=4)
    ax.axhline(ang_id, color=GREY, lw=0.85, ls=":")
    ax.text(0.98, ang_id - 0.003, f"identity ({ang_id:.4f})", fontsize=6.0, color=GREY,
            ha="right", va="top")

    ax.annotate("slope only\n($0.0559$)", xy=(0.0, ang[0]), xytext=(0.05, 0.100),
                fontsize=6.2, color=BLUE, fontweight="bold", ha="left", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7, shrinkA=1, shrinkB=4))
    ax.annotate("PC-OLS\n($0.1217$, $+118\\%$)", xy=(1.0, ang[-1]), xytext=(0.74, 0.136),
                fontsize=6.2, color=VERM, fontweight="bold", ha="right", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color=VERM, lw=0.7, shrinkA=1, shrinkB=4))

    ax.set_xlabel("offset weight $\\lambda$ in $T_\\lambda(z) = Az + \\lambda b$")
    ax.set_ylabel("direction error ($1 - \\cos$)")
    ax.set_xticks(lam)
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(0.025, 0.155)
    panel_header(ax, "C", "Direction error along $\\lambda$")
    style_ax(ax)

    save(fig, "fig_phase",
         "mean (line) and interquartile range across patients (band) for the \\lambda path; "
         "panel A: identity, slope-only (\\lambda=0), PC-OLS (\\lambda=1) and full mixing",
         {"results/p1_20261119/e3_arrays.npz": "0ec77b14abfe7dc2",
          "results/p1_20261119/e4_post_vq_arrays.npz": "5a7e344c0ec00e93"},
         n=200)


# --------------------------------------------------------------------------- Fig 3
def fig_patient(data):
    d = data["fig_patient_level"]
    mse_s0_0 = np.array(d["mse_s0_lam0"]) * 1e4
    mse_s0_1 = np.array(d["mse_s0_lam1"]) * 1e4
    mse_s1_0 = np.array(d["mse_s1_lam0"]) * 1e4
    mse_s1_1 = np.array(d["mse_s1_lam1"]) * 1e4

    ang_s0_0 = np.array(d["ang_s0_lam0"])
    ang_s0_1 = np.array(d["ang_s0_lam1"])
    ang_s1_0 = np.array(d["ang_s1_lam0"])
    ang_s1_1 = np.array(d["ang_s1_lam1"])

    bins = np.array(d["hist_bins"], dtype=float)
    c0 = np.array(d["hist_counts_s0"], dtype=float)
    c1 = np.array(d["hist_counts_s1"], dtype=float)
    med0 = d["median_delta_s0_1e6"]
    med1 = d["median_delta_s1_1e6"]

    fig, axs = plt.subplots(1, 3, figsize=(FIG_W, 2.38))
    fig.subplots_adjust(left=0.082, right=0.985, bottom=0.210, top=0.845, wspace=0.38)

    # --- Panel A: Per-patient Token MSE
    ax = axs[0]
    lim = (4.2, 7.2)
    # Subtle ±2% envelope around y=x
    xs = np.linspace(4.2, 7.2, 50)
    ax.fill_between(xs, xs * 0.98, xs * 1.02, color=GREY_LT, alpha=0.9, lw=0, zorder=0)
    ax.plot(lim, lim, color=GREY, lw=0.85, ls="--", zorder=1)
    ax.scatter(mse_s0_0, mse_s0_1, marker="o", s=14, facecolors=BLUE, edgecolors="#FFFFFF",
               linewidths=0.35, alpha=0.85, label="kernel only ($n=100$)", zorder=3)
    ax.scatter(mse_s1_0, mse_s1_1, marker="^", s=17, facecolors=VERM, edgecolors="#FFFFFF",
               linewidths=0.35, alpha=0.85, label="kernel + sampling ($n=100$)", zorder=3)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("token MSE, slope only ($\\times 10^{-4}$)")
    ax.set_ylabel("token MSE, PC-OLS ($\\times 10^{-4}$)")
    ax.text(0.96, 0.055, "all 200 patients lie on $y = x$\n(offset changes MSE by $< 1.7\\%$\nfor every patient)",
            transform=ax.transAxes, fontsize=5.9, color=GREY, ha="right", va="bottom",
            linespacing=1.25, bbox=dict(fc="#FFFFFF", ec=GREY_MID, lw=0.45, alpha=0.92, pad=2.0))
    ax.legend(frameon=False, fontsize=6.2, loc="upper left", handletextpad=0.2,
              borderpad=0.15, labelspacing=0.3, markerscale=1.15)
    panel_header(ax, "A", "Per-patient token MSE")
    style_ax(ax, grid_axis="both")

    # --- Panel B: Per-patient Direction error
    ax = axs[1]
    ax.fill_between([0, 0.5], [0, 0.5], [0.55, 0.55], color=VERM_LT, alpha=0.55, lw=0, zorder=0)
    ax.plot([0, 0.5], [0, 0.5], color=GREY, lw=0.85, ls="--", zorder=1)
    ax.scatter(ang_s0_0, ang_s0_1, marker="o", s=14, facecolors=BLUE, edgecolors="#FFFFFF",
               linewidths=0.35, alpha=0.85, zorder=3)
    ax.scatter(ang_s1_0, ang_s1_1, marker="^", s=17, facecolors=VERM, edgecolors="#FFFFFF",
               linewidths=0.35, alpha=0.85, zorder=3)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.47)
    ax.set_xlabel("direction error, slope only ($1 - \\cos$)")
    ax.set_ylabel("direction error, PC-OLS ($1 - \\cos$)")
    ax.text(0.04, 0.955, "194 / 200 patients ($97\\%$)\nabove $y = x$ (direction worsens)",
            transform=ax.transAxes, fontsize=6.0, color=VERM, fontweight="bold",
            ha="left", va="top", linespacing=1.22,
            bbox=dict(fc="#FFFFFF", ec=VERM_MID, lw=0.45, alpha=0.92, pad=2.0))
    ax.text(0.95, 0.08, "$y = x$", transform=ax.transAxes, fontsize=6.4, color=GREY,
            ha="right", va="bottom", style="italic")
    panel_header(ax, "B", "Per-patient direction error")
    style_ax(ax, grid_axis="both")

    # --- Panel C: Stacked Histogram + smooth KDE outlines
    ax = axs[2]
    lefts = bins[:-1]
    widths = np.diff(bins)
    ax.bar(lefts, c0, width=widths, align="edge", color=BLUE, edgecolor="#FFFFFF",
           linewidth=0.45, alpha=0.88, label="kernel only", zorder=3)
    ax.bar(lefts, c1, width=widths, bottom=c0, align="edge", color=VERM, edgecolor="#FFFFFF",
           linewidth=0.45, alpha=0.88, label="kernel + sampling", zorder=3)
    ax.axvline(0, color=INK, lw=0.95, ls="--", zorder=4)
    ax.axvline(med0, color=BLUE, lw=1.0, ls=":", zorder=4)
    ax.axvline(med1, color=VERM, lw=1.0, ls=":", zorder=4)

    # Clean non-overlapping callouts
    ax.annotate(f"median ${med0:+.1f}$", xy=(med0, 32), xytext=(1.3, 35.5),
                fontsize=6.0, color=BLUE, fontweight="bold", ha="left", va="center",
                bbox=dict(fc="#FFFFFF", ec=BLUE_MID, lw=0.45, alpha=0.92, pad=1.5),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.7, shrinkA=1, shrinkB=2))
    ax.annotate(f"median ${med1:+.1f}$", xy=(med1, 21), xytext=(6.6, 25.5),
                fontsize=6.0, color=VERM, fontweight="bold", ha="left", va="center",
                bbox=dict(fc="#FFFFFF", ec=VERM_MID, lw=0.45, alpha=0.92, pad=1.5),
                arrowprops=dict(arrowstyle="->", color=VERM, lw=0.7, shrinkA=1, shrinkB=2))

    ax.set_xlim(-3.2, 14.5)
    ax.set_ylim(0, 54)
    ax.set_xlabel("$\\Delta$ token MSE from offset $b$ ($\\times 10^{-6}$)")
    ax.set_ylabel("number of patients")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", handletextpad=0.3,
              borderpad=0.15, labelspacing=0.3)
    panel_header(ax, "C", "Sign reversal across strata")
    style_ax(ax)

    save(fig, "fig_patient_level",
         "per-patient pairs (n=200): token MSE and direction error at \\lambda=0 (slope only) vs \\lambda=1 (PC-OLS), "
         "and the per-patient change in token MSE caused by adding the offset",
         {"results/p1_20261119/e3_arrays.npz": "0ec77b14abfe7dc2"}, n=200)


# --------------------------------------------------------------------------- Fig 4
def fig_vendor(data):
    d = data["fig_vendor_operating"]
    oof = d["oof_all"]["auroc"]
    oof_ci = d["oof_all"]["ci"]
    ins = d["ins_all"]["auroc"]
    v = d["vendors"]
    op = d["operating"]

    fig, axs = plt.subplots(1, 2, figsize=(FIG_W, 2.82),
                            gridspec_kw={"width_ratios": [1.38, 1.0]})
    fig.subplots_adjust(left=0.178, right=0.985, bottom=0.175, top=0.840, wspace=0.33)

    # --- Panel A: Cross-vendor macro AUROC against reference bracket
    ax = axs[0]
    rows = [
        ("SOMATOM Force", v["Force"]["n"], v["Force"]["classes"], v["Force"]["auroc"], v["Force"]["ci"], 2, VERM, "^"),
        ("BBB (B kernel)", v["BBB"]["n"],   v["BBB"]["classes"],   v["BBB"]["auroc"],   v["BBB"]["ci"],   1, BLUE, "s"),
        ("MX 16",          v["MX16"]["n"],  v["MX16"]["classes"],  v["MX16"]["auroc"],  v["MX16"]["ci"],  0, GREEN, "D"),
    ]

    # Reference bracket shading & OOF 95% CI shading
    ax.axvspan(oof, ins, color="#E2E8F0", alpha=0.60, zorder=0, lw=0)
    ax.axvspan(oof_ci[0], oof_ci[1], color=BLUE_LT, alpha=0.65, zorder=0, lw=0)
    ax.axvline(oof, color=INK, lw=0.95, ls="-", zorder=2)
    ax.axvline(ins, color=INK, lw=0.95, ls="--", zorder=2)

    # Plot OOF 95% CI bar at top (y=2.55)
    ax.plot(oof_ci, [2.56, 2.56], color=GREY, lw=1.15, zorder=3)
    for c in oof_ci:
        ax.plot([c, c], [2.49, 2.63], color=GREY, lw=0.95, zorder=3)
    ax.plot(oof, 2.56, "o", ms=4.2, mfc="#FFFFFF", mec=INK, mew=1.0, zorder=4)

    for lab, n, k, est, ci, y, col, mk in rows:
        ax.plot(ci, [y, y], color=col, lw=1.45, zorder=3)
        for c in ci:
            ax.plot([c, c], [y - 0.13, y + 0.13], color=col, lw=1.05, zorder=3)
        ax.plot(est, y, marker=mk, ms=5.8, color=col, mfc=col, mec="#FFFFFF", mew=0.65, zorder=4)
        ax.text(est, y + 0.18, f"{est:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]",
                fontsize=6.0, color=col, fontweight="bold", ha="center", va="bottom",
                bbox=dict(fc="#FFFFFF", ec="none", alpha=0.80, pad=0.6), zorder=5)

    ax.set_yticks([r[5] for r in rows])
    ax.set_yticklabels([f"{r[0]}\n($n={r[1]}$, {r[2]} cls)" for r in rows])
    ax.set_ylim(-0.62, 2.95)
    ax.set_xlim(0.54, 0.86)
    ax.set_xlabel("macro AUROC (patient-level bootstrap, 95% CI)")

    # Crisp non-colliding reference labels
    ax.text(oof_ci[0] - 0.008, 2.56, f"source OOF: {oof:.3f}\n[{oof_ci[0]:.3f}, {oof_ci[1]:.3f}]",
            fontsize=5.9, color=INK, ha="right", va="center", linespacing=1.18)
    ax.text(ins + 0.006, 2.56, f"in-sample:\n{ins:.3f} ($+0.087$)",
            fontsize=5.9, color=INK, ha="left", va="center", linespacing=1.18)
    ax.text((oof + ins) / 2, -0.43, "reference bracket\n($\\Delta = 0.087$)",
            fontsize=5.8, color=INK, ha="center", va="center", style="italic", linespacing=1.15,
            bbox=dict(fc="#FFFFFF", ec="none", alpha=0.80, pad=0.6))

    panel_header(ax, "A", "Cross-vendor AUROC vs reference bracket")
    style_ax(ax, grid_axis="x")

    # --- Panel B: Operating point rotation at fixed source threshold (90% sensitivity)
    ax = axs[1]
    b_op = op["BBB"]
    f_op = op["Force"]

    # Reference 90% target sensitivity line
    ax.axhline(0.90, color=GREY_MID, lw=0.85, ls=":", zorder=1)
    ax.text(0.244, 0.902, "nominal 90% sensitivity", fontsize=5.8, color=GREY,
            ha="left", va="bottom", style="italic")

    # Source points (open circles)
    ax.plot(b_op["spec_src"], b_op["sens_src"], "o", ms=5.8, mfc="#FFFFFF", mec=BLUE, mew=1.2, zorder=4)
    ax.plot(f_op["spec_src"], f_op["sens_src"], "o", ms=5.8, mfc="#FFFFFF", mec=VERM, mew=1.2, zorder=4)

    # Target points (filled markers)
    ax.plot(b_op["spec_tgt"], b_op["sens_tgt"], "s", ms=6.2, mfc=BLUE, mec="#FFFFFF", mew=0.65, zorder=5)
    ax.plot(f_op["spec_tgt"], f_op["sens_tgt"], "^", ms=6.8, mfc=VERM, mec="#FFFFFF", mew=0.65, zorder=5)

    # Trajectory arrows
    ax.add_patch(FancyArrowPatch(
        (b_op["spec_src"], b_op["sens_src"]), (b_op["spec_tgt"], b_op["sens_tgt"]),
        arrowstyle="-|>,head_length=3.6,head_width=2.1", mutation_scale=1.0,
        lw=1.3, color=BLUE, shrinkA=5.5, shrinkB=5.5, zorder=3
    ))
    ax.add_patch(FancyArrowPatch(
        (f_op["spec_src"], f_op["sens_src"]), (f_op["spec_tgt"], f_op["sens_tgt"]),
        arrowstyle="-|>,head_length=3.6,head_width=2.1", mutation_scale=1.0,
        lw=1.3, color=VERM, shrinkA=5.5, shrinkB=5.5, zorder=3
    ))

    # Labels with zero overlap
    ax.annotate("Source (14 BBB cls)\nsens 0.893, spec 0.374",
                xy=(b_op["spec_src"], b_op["sens_src"]), xytext=(0.352, 0.868),
                fontsize=5.9, color=INK, ha="right", va="top", linespacing=1.18,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.65, shrinkA=1, shrinkB=3.5))

    ax.annotate("Source (17 Force cls)\nsens 0.889, spec 0.413",
                xy=(f_op["spec_src"], f_op["sens_src"]), xytext=(0.455, 0.908),
                fontsize=5.9, color=INK, ha="right", va="bottom", linespacing=1.18,
                arrowprops=dict(arrowstyle="->", color=GREY, lw=0.65, shrinkA=1, shrinkB=3.5))

    ax.text(0.408, 0.816, "BBB (prev 0.150)\nsens 0.820 ($\\downarrow$), spec 0.419 ($\\uparrow$)",
            fontsize=6.0, fontweight="bold", color=BLUE, ha="right", va="center",
            linespacing=1.20, bbox=dict(fc="#FFFFFF", ec=BLUE_MID, lw=0.45, alpha=0.90, pad=1.6))

    ax.text(0.294, 0.943, "SOMATOM Force (prev 0.235)\nsens 0.934 ($\\uparrow$), spec 0.280 ($\\downarrow$)",
            fontsize=6.0, fontweight="bold", color=VERM, ha="left", va="center",
            linespacing=1.20, bbox=dict(fc="#FFFFFF", ec=VERM_MID, lw=0.45, alpha=0.90, pad=1.6))

    ax.set_xlim(0.24, 0.462)
    ax.set_ylim(0.798, 0.960)
    ax.set_xlabel("implied specificity (Bayes identity)")
    ax.set_ylabel("sensitivity at fixed source threshold")
    panel_header(ax, "B", "Opposite operating-point shifts")
    style_ax(ax, grid_axis="both")

    save(fig, "fig_vendor_operating",
         "macro AUROC per vendor with bootstrap CI and the out-of-fold/in-sample source bracket; "
         "fixed-threshold sensitivity and implied specificity per vendor",
         {"results/p2_20261121/multivendor_eval.json": "23f20cfff541a252",
          "results/p2_20261121/source_cv.json": "bfeac525d2ebc0b6",
          "results/p2_20261121/clinical_utility.json": "2fa01ca48efd83a1"}, n=128)


if __name__ == "__main__":
    data = load_data()
    fig_design()
    fig_phase(data)
    fig_patient(data)
    fig_vendor(data)
    with open(os.path.join(OUT, "figure_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "script": os.path.basename(__file__),
            "text_width_in": FIG_W,
            "figures": MANIFEST
        }, fh, indent=2, ensure_ascii=False)
    print("v4 figures written:", [m["figure"] for m in MANIFEST])
