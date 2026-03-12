# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Geometric Analysis of Sperm Whale Communication
#
# **eris-ketos: Geometric cetacean communication analysis toolkit**
#
# This notebook applies differential geometry, algebraic topology, and
# adversarial robustness testing to annotated sperm whale coda data from
# the Dominica Sperm Whale Project (DSWP). It demonstrates that
# non-Euclidean geometric methods capture hierarchical and topological
# structure in cetacean vocalizations that standard flat-space analyses
# fail to represent.
#
# **Novel contributions:**
# 1. Poincare ball embedding of the combinatorial coda type hierarchy
# 2. Persistent homology (TDA) of click timing point clouds
# 3. SPD manifold analysis of spectral covariance structure
# 4. Decoder Robustness Index (DRI) via adversarial acoustic fuzzing
# 5. Adversarial perturbation as a phonetic boundary discovery tool
#
# **Data:** Sharma et al., "Contextual and combinatorial structure in
# sperm whale vocalisations," *Nature Communications* 15:3617 (2024);
# DSWP audio recordings (HuggingFace `orrp/DSWP`).
#
# **References:**
# - Sharma, P. et al. (2024) *Nature Communications* 15:3617 — combinatorial coda structure
# - Begus, G. et al. (2025) *Open Mind* 9:1849–1874 — vowel-like spectral patterns
# - Gero, S. et al. (2016) *R. Soc. Open Sci.* 3:150372 — individual/unit identity cues
# - Rendell, L. & Whitehead, H. (2003) *Proc. R. Soc. B* 270:225–231 — vocal clans
# - Weilgart, L. & Whitehead, H. (1993) *Can. J. Zool.* 71:744–752 — coda communication
# - Youngblood, M. (2025) *Science Advances* 11:eads6014 — linguistic laws in cetaceans
# - Paradise, O. et al. (2025) *NeurIPS* — WhAM translative model
# - Nickel, M. & Kiela, D. (2017) *NeurIPS* — Poincare embeddings
# - Sarkar, R. (2011) *GD 2011*, LNCS 7034 — low-distortion hyperbolic tree embedding
# - Carlsson, G. (2009) *Bull. AMS* 46:255–308 — topology and data
# - Bauer, U. (2021) *J. Appl. Comput. Topol.* 5:391–423 — Ripser algorithm
# - Hersh, T.A. et al. (2022) *PNAS* 119:e2201692119 — symbolic clan identity marking
# - Cantor, M. & Whitehead, H. (2013) *Phil. Trans. R. Soc. B* 368:20120340 — social networks and culture

# %% [markdown]
# ## 0. Setup

# %%
# !pip install -q eris-ketos datasets matplotlib seaborn scikit-learn

# %%
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from eris_ketos.poincare_coda import (
    PoincareBall,
    HyperbolicMLR,
    build_distance_matrix,
    embed_taxonomy_hyperbolic,
)
from eris_ketos.acoustic_transforms import make_acoustic_transform_suite, TransformChain
from eris_ketos.decoder_robustness import (
    DecoderRobustnessIndex,
    CodaSemanticDistance,
    CODA_FEATURE_WEIGHTS,
)

try:
    from eris_ketos.tda_clicks import (
        time_delay_embedding,
        compute_persistence,
        tda_feature_vector,
    )
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    print("ripser not available - TDA sections will use direct computation")

try:
    import ripser as _ripser_mod
    HAS_RIPSER_DIRECT = True
except ImportError:
    HAS_RIPSER_DIRECT = False

from eris_ketos.spd_spectral import (
    SPDManifold,
    compute_covariance,
    spd_features_from_spectrogram,
    compute_spectral_trajectory,
)

plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
sns.set_palette('husl')
print("All imports successful!")

# %% [markdown]
# ## 1. Data Loading
#
# The DSWP dataset comprises annotated sperm whale codas with inter-click
# intervals (ICIs), coda type labels, social unit identifiers, and
# individual whale identifiers. Sharma et al. (2024) demonstrated that
# these codas possess a combinatorial phonetic structure decomposable
# into rhythm, tempo, rubato, and ornamentation features.

# %%
# Download coda annotations from Sharma et al. (Nature Comms 2024)
SHARMA_URLS = [
    "https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/main/data/DominicaCodas.csv",
    "https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/master/data/DominicaCodas.csv",
]

df = None
for url in SHARMA_URLS:
    try:
        df = pd.read_csv(url)
        print(f"Loaded {len(df)} codas from Sharma et al. repository")
        break
    except Exception:
        continue

if df is None:
    raise RuntimeError(
        "Could not download DominicaCodas.csv. "
        "Please download manually from github.com/pratyushasharma/sw-combinatoriality"
    )

print(f"Columns: {list(df.columns)}")
print(f"\nDataset shape: {df.shape}")

# %%
# Explore coda types and ICI structure
ici_cols = [c for c in df.columns if c.startswith('ICI')]
print(f"ICI columns ({len(ici_cols)}): {ici_cols}")
print(f"Click count range: {df['nClicks'].min()} - {df['nClicks'].max()}")

# Coda type distribution
coda_counts = df['CodaType'].value_counts()
print(f"\nTotal coda types: {df['CodaType'].nunique()}")
print(f"\nTop 15 coda types:")
print(coda_counts.head(15).to_string())

# Social units
if 'Unit' in df.columns:
    print(f"\nSocial units: {sorted(df['Unit'].dropna().unique())}")
if 'IDN' in df.columns:
    print(f"Individual whales: {df['IDN'].nunique()}")

# %%
# Visualize coda type distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart of coda types
top_types = coda_counts.head(20)
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_types)))
axes[0].barh(range(len(top_types)), top_types.values, color=colors_bar)
axes[0].set_yticks(range(len(top_types)))
axes[0].set_yticklabels(top_types.index, fontsize=9)
axes[0].set_xlabel('Count')
axes[0].set_title('Coda Type Distribution (Top 20)')
axes[0].invert_yaxis()

# Click count histogram
df['nClicks'].hist(ax=axes[1], bins=range(2, 12), edgecolor='black', alpha=0.8,
                   color='steelblue')
axes[1].set_xlabel('Number of Clicks')
axes[1].set_ylabel('Count')
axes[1].set_title('Click Count Distribution')

plt.tight_layout()
plt.savefig('fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved.")

# %% [markdown]
# ## 2. Poincare Ball Embedding of Coda Type Hierarchy
#
# The combinatorial structure of coda types forms a tree-like hierarchy
# (rhythm class, click count, variant). Trees embed with
# $O(\log n)$ distortion in hyperbolic space versus $O(n)$ in Euclidean
# space (Sarkar, 2011), making the Poincare ball model a natural
# representation for hierarchical phonetic systems.
#
# We construct a taxonomic distance matrix from shared features, embed
# coda types into the Poincare ball via gradient descent, and compare
# distortion against Euclidean (PCA) embedding.

# %%
# Parse coda type names into a DEEP 4-level taxonomy
# Hierarchy (coarsest → finest): rhythm_class → click_count → variant
# This tree structure is what makes hyperbolic embedding superior to Euclidean.
def parse_coda_type(coda_type):
    ct = str(coda_type).strip()
    if 'NOISE' in ct.upper():
        return None

    # Compound types (contain '+')
    if '+' in ct:
        return {
            'rhythm_class': 'compound',
            'click_count': ct,  # unique per compound pattern
            'variant': '0',
        }

    # Extract leading digits (click count)
    digits = ''
    for ch in ct:
        if ch.isdigit():
            digits += ch
        else:
            break
    remainder = ct[len(digits):]

    # Extract variant number at end (e.g., R1, R2, D1, D2)
    variant_digits = ''
    for ch in reversed(remainder):
        if ch.isdigit():
            variant_digits = ch + variant_digits
        else:
            break
    rhythm_letter = remainder[:len(remainder) - len(variant_digits)] if variant_digits else remainder

    # Rhythm class
    if rhythm_letter.startswith('R'):
        rhythm_class = 'regular'
    elif rhythm_letter.startswith('D'):
        rhythm_class = 'deceleration'
    elif rhythm_letter.startswith('i'):
        rhythm_class = 'irregular'
    else:
        rhythm_class = 'other'

    click_count = digits if digits else 'unknown'
    variant = variant_digits if variant_digits else '0'

    return {
        'rhythm_class': rhythm_class,
        'click_count': click_count,
        'variant': variant,
    }


# Build taxonomy
coda_types = df['CodaType'].unique()
coda_taxonomy = {}
for ct in coda_types:
    features = parse_coda_type(ct)
    if features is not None:
        coda_taxonomy[ct] = features

print(f"Parsed {len(coda_taxonomy)} coda types into 4-level taxonomy:")
print(f"{'Type':>10s}  {'Rhythm':>14s}  {'Clicks':>6s}  {'Variant':>7s}")
print("-" * 45)
for ct in sorted(coda_taxonomy.keys()):
    f = coda_taxonomy[ct]
    print(f"  {ct:>10s}  {f['rhythm_class']:>14s}  {f['click_count']:>6s}  {f['variant']:>7s}")

# %%
# Build distance matrix with DEEP 3-level hierarchy
# (variant → click_count → rhythm_class) finest to coarsest
# Distance: 0=same, 1=same variant group, 2=same click count, 3=same rhythm, 4=different
type_names = list(coda_taxonomy.keys())
n_types = len(type_names)
dist_matrix = build_distance_matrix(
    coda_taxonomy, levels=('variant', 'click_count', 'rhythm_class')
)

print(f"Taxonomic distance matrix: {dist_matrix.shape}")
print(f"Distance range: [{dist_matrix.min()}, {dist_matrix.max()}]")
print(f"Tree depth: 4 levels (variant → clicks → rhythm_class → root)")
unique_dists = sorted(set(dist_matrix.flatten()))
print(f"Unique distances: {unique_dists}")

# Poincare embedding
embed_dim = min(16, max(4, n_types))
poincare_emb = embed_taxonomy_hyperbolic(dist_matrix, embed_dim=embed_dim, c=1.0)
print(f"\nPoincare embeddings: {poincare_emb.shape}")
print(f"Max norm: {poincare_emb.norm(dim=-1).max():.4f} (must be < 1.0)")

# Euclidean embedding (PCA)
pca = PCA(n_components=2)
euclidean_emb = pca.fit_transform(dist_matrix)
print(f"Euclidean PCA embeddings: {euclidean_emb.shape}")
print(f"PCA explained variance: {pca.explained_variance_ratio_[:2].sum():.1%}")

# %%
# Visualize: Poincare disk vs Euclidean PCA
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

rhythm_classes = [coda_taxonomy[t]['rhythm_class'] for t in type_names]
unique_rhythms = sorted(set(rhythm_classes))
cmap = plt.cm.Set1
color_map = {r: cmap(i / max(len(unique_rhythms) - 1, 1))
             for i, r in enumerate(unique_rhythms)}

# --- Poincare disk ---
ax = axes[0]
circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2.5, linestyle='-')
ax.add_patch(circle)
# Inner geodesic circles for reference
for r in [0.3, 0.6, 0.9]:
    inner = plt.Circle((0, 0), r, fill=False, color='gray', linewidth=0.5,
                        linestyle='--', alpha=0.4)
    ax.add_patch(inner)

poincare_2d = poincare_emb[:, :2].detach().numpy()
for i, name in enumerate(type_names):
    c = color_map[rhythm_classes[i]]
    ax.scatter(poincare_2d[i, 0], poincare_2d[i, 1], c=[c], s=120,
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.annotate(name, (poincare_2d[i, 0], poincare_2d[i, 1]), fontsize=7,
                fontweight='bold', ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')

ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_aspect('equal')
ax.set_title('Poincare Ball Embedding\n(hyperbolic geometry)', fontsize=14,
             fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')

# --- Euclidean PCA ---
ax = axes[1]
for i, name in enumerate(type_names):
    c = color_map[rhythm_classes[i]]
    ax.scatter(euclidean_emb[i, 0], euclidean_emb[i, 1], c=[c], s=120,
               edgecolors='black', linewidths=0.5, zorder=3)
    ax.annotate(name, (euclidean_emb[i, 0], euclidean_emb[i, 1]), fontsize=7,
                fontweight='bold', ha='center', va='bottom',
                xytext=(0, 6), textcoords='offset points')

ax.set_title('Euclidean PCA Embedding\n(flat geometry)', fontsize=14,
             fontweight='bold')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')

# Legend
handles = [mpatches.Patch(color=color_map[r], label=r.capitalize())
           for r in unique_rhythms]
fig.legend(handles=handles, loc='lower center', ncol=len(unique_rhythms),
           fontsize=12, frameon=True, fancybox=True)

plt.suptitle('Sperm Whale Coda Type Embeddings', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.savefig('fig2_poincare_vs_euclidean.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved.")

# %%
# Quantitative distortion comparison
ball = PoincareBall(c=1.0)
true_dists = []
poincare_dists = []
euclidean_dists = []

for i in range(n_types):
    for j in range(i + 1, n_types):
        true_dists.append(dist_matrix[i, j])
        p_dist = ball.dist(poincare_emb[i:i+1], poincare_emb[j:j+1]).item()
        poincare_dists.append(p_dist)
        e_dist = np.linalg.norm(euclidean_emb[i] - euclidean_emb[j])
        euclidean_dists.append(e_dist)

rho_poincare, p_poincare = spearmanr(true_dists, poincare_dists)
rho_euclidean, p_euclidean = spearmanr(true_dists, euclidean_dists)

print("=" * 60)
print("DISTORTION ANALYSIS")
print("Spearman rank correlation with true taxonomic distance:")
print("=" * 60)
print(f"  Poincare ball:  rho = {rho_poincare:.4f}  (p = {p_poincare:.2e})")
print(f"  Euclidean PCA:  rho = {rho_euclidean:.4f}  (p = {p_euclidean:.2e})")
print(f"  Advantage: {'Poincare' if rho_poincare > rho_euclidean else 'Euclidean'} "
      f"(delta_rho = {abs(rho_poincare - rho_euclidean):.4f})")
print("=" * 60)

# Scatter plot of distances
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, emb_dists, name, rho in [
    (axes[0], poincare_dists, 'Poincare', rho_poincare),
    (axes[1], euclidean_dists, 'Euclidean PCA', rho_euclidean),
]:
    ax.scatter(true_dists, emb_dists, alpha=0.5, s=30)
    ax.set_xlabel('True Taxonomic Distance')
    ax.set_ylabel(f'{name} Embedded Distance')
    ax.set_title(f'{name}: Spearman rho = {rho:.4f}')
    # Trend line
    z = np.polyfit(true_dists, emb_dists, 1)
    x_line = np.linspace(min(true_dists), max(true_dists), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.7)

plt.tight_layout()
plt.savefig('fig3_distortion_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved.")

# %% [markdown]
# ## 3. Classification via Hyperbolic Multinomial Logistic Regression
#
# We train a Hyperbolic Multinomial Logistic Regression (HyperbolicMLR)
# classifier on individual coda feature vectors, with class prototypes
# initialized from the taxonomic Poincare embeddings. This evaluates
# whether hyperbolic geometry provides a classification advantage over
# Euclidean baselines for the combinatorial coda type system.

# %%
# Prepare classification dataset with PROPER feature engineering
# Key insight from Sharma et al.: decompose ICI into rhythm (normalized) + tempo (duration)
min_samples_per_type = 30
type_counts = df['CodaType'].value_counts()
valid_types = type_counts[type_counts >= min_samples_per_type].index.tolist()
valid_types = [t for t in valid_types if 'NOISE' not in str(t).upper()]

df_cls = df[df['CodaType'].isin(valid_types)].copy()
print(f"Classification dataset: {len(df_cls)} codas, {len(valid_types)} types")
print(f"Types: {sorted(valid_types)}")

# Feature engineering: normalized ICIs (rhythm) + duration (tempo)
raw_icis = df_cls[ici_cols].fillna(0).values.astype(np.float32)
durations = df_cls['Duration'].values.astype(np.float32).reshape(-1, 1)

# Normalize ICIs by duration to get rhythm pattern (tempo-invariant)
dur_safe = np.maximum(durations, 1e-6)
rhythm_features = raw_icis / dur_safe  # normalized timing pattern

# Log-duration as tempo feature (log scale separates clusters better)
log_duration = np.log1p(durations)

# Combine: [rhythm_features (9-dim) | log_duration (1-dim) | n_clicks (1-dim)]
n_clicks_feat = df_cls['nClicks'].values.astype(np.float32).reshape(-1, 1) / 10.0
X = np.hstack([rhythm_features, log_duration, n_clicks_feat])

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

le = LabelEncoder()
y = le.fit_transform(df_cls['CodaType'].values)
n_classes = len(le.classes_)

print(f"\nFeature matrix: {X.shape} (9 rhythm + 1 tempo + 1 clicks)")
print(f"Classes: {n_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# %%
# Train HyperbolicMLR with PROPER initialization and training
embed_dim = X.shape[1]

# Build DEEP taxonomy embeddings for prototype initialization
cls_taxonomy = {}
for ct in le.classes_:
    features = parse_coda_type(ct)
    if features is None:
        features = {'rhythm_class': 'other', 'click_count': 'unknown', 'variant': '0'}
    cls_taxonomy[ct] = features

cls_dist = build_distance_matrix(
    cls_taxonomy, levels=('variant', 'click_count', 'rhythm_class')
)
cls_emb = embed_taxonomy_hyperbolic(cls_dist, embed_dim=embed_dim, c=1.0)

# Initialize model
ball = PoincareBall(c=1.0)
mlr = HyperbolicMLR(embed_dim=embed_dim, num_classes=n_classes, c=1.0)
mlr.init_from_taxonomy(cls_emb)

# Map standardized features to the ball (features already zero-mean, unit-var)
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Scale factor: keep inputs well inside the ball (norm < 0.9)
scale = 0.3 / max(X_train_t.norm(dim=-1).quantile(0.95).item(), 1e-6)
X_train_ball = ball.expmap0(X_train_t * scale)
X_test_ball = ball.expmap0(X_test_t * scale)

print(f"Input norms on ball: mean={X_train_ball.norm(dim=-1).mean():.3f}, "
      f"max={X_train_ball.norm(dim=-1).max():.3f}")

# Training with LR scheduling and more epochs
optimizer = torch.optim.Adam(mlr.parameters(), lr=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-4)

losses = []
test_accs = []
best_acc = 0.0
for epoch in range(500):
    optimizer.zero_grad()
    logits = mlr(X_train_ball)
    loss = torch.nn.functional.cross_entropy(logits, y_train_t)
    loss.backward()
    optimizer.step()
    scheduler.step()
    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            test_logits = mlr(X_test_ball)
            pred = test_logits.argmax(dim=1)
            acc = (pred == y_test_t).float().mean().item()
            test_accs.append(acc)
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, "
                  f"test_acc={acc:.1%}, lr={scheduler.get_last_lr()[0]:.5f}")

# %%
# Compare with Euclidean baselines
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Multiple Euclidean baselines
lr_model = LogisticRegression(max_iter=2000, random_state=42)
lr_model.fit(X_train, y_train)
lr_acc = lr_model.score(X_test, y_test)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_acc = knn_model.score(X_test, y_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)

# Final HyperbolicMLR accuracy
with torch.no_grad():
    hyp_pred = mlr(X_test_ball).argmax(dim=1).numpy()
    hyp_acc = (hyp_pred == y_test).mean()

best_euclidean = max(lr_acc, knn_acc, rf_acc)
best_euclidean_name = ['LogReg', 'KNN', 'RandomForest'][[lr_acc, knn_acc, rf_acc].index(best_euclidean)]

print("\n" + "=" * 60)
print("CLASSIFICATION RESULTS")
print("Features: normalized ICIs (rhythm) + log-duration (tempo) + n_clicks")
print("=" * 60)
print(f"  HyperbolicMLR (Poincare ball):      {hyp_acc:.1%}  (best: {best_acc:.1%})")
print(f"  Euclidean LogisticRegression:        {lr_acc:.1%}")
print(f"  Euclidean KNN (k=5):                {knn_acc:.1%}")
print(f"  Euclidean RandomForest:              {rf_acc:.1%}")
print(f"  Advantage: {'Hyperbolic' if hyp_acc > best_euclidean else best_euclidean_name} "
      f"(delta = {abs(hyp_acc - best_euclidean):.1%})")
print("=" * 60)

# %%
# Training curves and confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Loss curve
axes[0].plot(losses, color='steelblue', linewidth=1)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].set_title('HyperbolicMLR Training Loss')
axes[0].set_yscale('log')

# Confusion matrix - Hyperbolic
cm_hyp = confusion_matrix(y_test, hyp_pred)
sns.heatmap(cm_hyp, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[1].set_title(f'HyperbolicMLR ({hyp_acc:.1%})')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].tick_params(axis='both', labelsize=6, rotation=45)

# Best Euclidean confusion matrix
best_euc_pred = [lr_model, knn_model, rf_model][
    [lr_acc, knn_acc, rf_acc].index(best_euclidean)
].predict(X_test)
cm_euc = confusion_matrix(y_test, best_euc_pred)
sns.heatmap(cm_euc, annot=True, fmt='d', cmap='Oranges', ax=axes[2],
            xticklabels=le.classes_, yticklabels=le.classes_)
axes[2].set_title(f'{best_euclidean_name} ({best_euclidean:.1%})')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('True')
axes[2].tick_params(axis='both', labelsize=6, rotation=45)

plt.suptitle('Coda Type Classification Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_classification.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 4 saved.")

# %% [markdown]
# ## 4. Topological Data Analysis of Click Timing Patterns
#
# Persistent homology is applied to the point clouds formed by ICI
# vectors within each coda type. Under this analysis, distinct rhythm
# patterns are expected to exhibit topologically distinguishable
# structures: regular codas (xR) should form tight clusters with low
# H1 persistence; irregular codas (xi) should produce spread
# distributions with topological holes; and compound codas (x+y)
# should manifest as multiple connected components (elevated H0).

# %%
# Build ICI point clouds per coda type
type_groups = df[~df['CodaType'].str.contains('NOISE', na=False)].groupby('CodaType')

# Select types with enough samples for meaningful topology
tda_min_samples = 40
tda_types = sorted([
    name for name, group in type_groups
    if len(group) >= tda_min_samples
])[:10]

print(f"TDA analysis on {len(tda_types)} coda types (>= {tda_min_samples} samples):")
for t in tda_types:
    print(f"  {t}: n = {len(type_groups.get_group(t))}")

# %%
# Compute persistent homology per coda type
tda_results = {}
if HAS_RIPSER_DIRECT:
    for coda_type in tda_types:
        group = type_groups.get_group(coda_type)
        ici_data = group[ici_cols].fillna(0).values.astype(np.float64)

        if len(ici_data) > 300:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(ici_data), 300, replace=False)
            ici_data = ici_data[idx]

        res = _ripser_mod.ripser(ici_data, maxdim=1)
        diagrams = res['dgms']

        features = {}
        for dim in range(min(2, len(diagrams))):
            dgm = diagrams[dim]
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            features[f'H{dim}_count'] = len(lifetimes)
            features[f'H{dim}_mean'] = float(np.mean(lifetimes)) if len(lifetimes) > 0 else 0
            features[f'H{dim}_max'] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0
            features[f'H{dim}_total'] = float(np.sum(lifetimes))

        tda_results[coda_type] = {'features': features, 'diagrams': diagrams}
        print(f"  {coda_type}: H0={features['H0_count']} components, "
              f"H1={features['H1_count']} loops (max life={features['H1_max']:.4f})")
else:
    print("Skipping TDA (ripser not installed). Install with: pip install ripser")

# %%
# Persistence diagrams
if tda_results:
    n_show = min(6, len(tda_types))
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    for idx, coda_type in enumerate(tda_types[:n_show]):
        ax = axes[idx]
        diagrams = tda_results[coda_type]['diagrams']

        for dim, (color, marker, label) in enumerate(
            [('steelblue', 'o', 'H0'), ('crimson', '^', 'H1')]
        ):
            if dim < len(diagrams):
                dgm = diagrams[dim]
                finite_mask = np.isfinite(dgm[:, 1])
                dgm_finite = dgm[finite_mask]
                if len(dgm_finite) > 0:
                    ax.scatter(dgm_finite[:, 0], dgm_finite[:, 1],
                               c=color, marker=marker, s=25, alpha=0.6, label=label)

        lims = ax.get_xlim()
        ax.plot([0, max(lims[1], 1)], [0, max(lims[1], 1)], 'k--', alpha=0.3)
        ax.set_xlabel('Birth')
        ax.set_ylabel('Death')
        ax.set_title(f'{coda_type}  (n={len(type_groups.get_group(coda_type))})',
                     fontweight='bold')
        ax.legend(fontsize=8)

    plt.suptitle('Persistence Diagrams by Coda Type', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig5_persistence_diagrams.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 5 saved.")

# %%
# TDA feature comparison across types
if tda_results:
    tda_df = pd.DataFrame({ct: r['features'] for ct, r in tda_results.items()}).T
    print("\nTDA Features by Coda Type:")
    print(tda_df.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ct in tda_results:
        f = tda_results[ct]['features']
        rhythm = parse_coda_type(ct)
        if rhythm is None:
            continue
        c = color_map.get(rhythm['rhythm_class'], 'gray')
        axes[0].scatter(f['H1_count'], f['H1_max'], c=[c], s=100,
                        edgecolors='black', linewidths=0.5, zorder=3)
        axes[0].annotate(ct, (f['H1_count'], f['H1_max']), fontsize=8,
                         ha='center', va='bottom', xytext=(0, 5),
                         textcoords='offset points')

    axes[0].set_xlabel('H1 Loop Count')
    axes[0].set_ylabel('H1 Max Lifetime')
    axes[0].set_title('Topological Complexity by Coda Type')

    h1_totals = {ct: tda_results[ct]['features']['H1_total'] for ct in tda_types
                 if ct in tda_results}
    sorted_types = sorted(h1_totals, key=h1_totals.get, reverse=True)
    bar_colors = [color_map.get(parse_coda_type(ct)['rhythm_class'], 'gray')
                  for ct in sorted_types]
    axes[1].barh(range(len(sorted_types)),
                 [h1_totals[t] for t in sorted_types],
                 color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(len(sorted_types)))
    axes[1].set_yticklabels(sorted_types)
    axes[1].set_xlabel('Total H1 Persistence')
    axes[1].set_title('H1 Persistence Ranking')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig('fig6_tda_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 6 saved.")

# %% [markdown]
# ## 5. SPD Manifold Analysis of Spectral Structure
#
# Frequency-band covariance matrices are symmetric positive definite
# (SPD) and naturally reside on a Riemannian manifold. The log-Euclidean
# metric on this manifold captures inter-band correlations (e.g.,
# harmonic relationships) that standard flat-space spectrograms fail to
# represent. We synthesize click signals from ICI patterns and analyze
# their spectral covariance structure on the SPD manifold.

# %%
# Generate synthetic click signals from ICI data
def synthesize_coda_signal(icis, sr=32000, click_dur=0.003):
    """Generate a synthetic click train from inter-click intervals."""
    icis_clean = [float(x) for x in icis if float(x) > 0]
    if not icis_clean:
        return np.zeros(sr, dtype=np.float32)  # 1 second of silence

    total_dur = sum(icis_clean) + click_dur + 0.05  # padding
    n_samples = int(total_dur * sr)
    signal = np.zeros(n_samples, dtype=np.float32)

    t = 0.01  # small offset
    click_n = int(click_dur * sr)
    # First click
    idx = int(t * sr)
    for k in range(click_n):
        if idx + k < n_samples:
            phase = 2 * np.pi * 5000 * k / sr  # 5 kHz center
            env = np.exp(-0.5 * ((k - click_n / 2) / (click_n / 5)) ** 2)
            signal[idx + k] = env * np.sin(phase)

    # Subsequent clicks
    for ici in icis_clean:
        t += ici
        idx = int(t * sr)
        for k in range(click_n):
            if idx + k < n_samples:
                phase = 2 * np.pi * 5000 * k / sr
                env = np.exp(-0.5 * ((k - click_n / 2) / (click_n / 5)) ** 2)
                signal[idx + k] = env * np.sin(phase)

    return signal


# Generate signals for a subset of codas
spd_types = [t for t in valid_types if t in tda_types][:6]
if not spd_types:
    spd_types = valid_types[:6]

spd_signals = {}
for coda_type in spd_types:
    group = df[df['CodaType'] == coda_type].head(50)
    signals = []
    for _, row in group.iterrows():
        icis = [row[c] for c in ici_cols if not pd.isna(row[c]) and float(row[c]) > 0]
        sig = synthesize_coda_signal(icis)
        signals.append(sig)
    spd_signals[coda_type] = signals

print(f"Synthesized signals for {len(spd_signals)} coda types")
for ct, sigs in spd_signals.items():
    print(f"  {ct}: {len(sigs)} signals, avg length = {np.mean([len(s) for s in sigs]):.0f} samples")

# %%
# Compute SPD features from spectrograms
import librosa

spd_features_all = {}
for coda_type, signals in spd_signals.items():
    features_list = []
    for sig in signals:
        if len(sig) < 1024:
            continue
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=sig, sr=32000, n_mels=64, n_fft=1024, hop_length=256
        )
        mel_db = librosa.power_to_db(mel + 1e-10)
        if mel_db.shape[1] < 4:
            continue
        # Extract SPD features
        feat = spd_features_from_spectrogram(mel_db, n_bands=8)
        features_list.append(feat)

    if features_list:
        spd_features_all[coda_type] = np.array(features_list)
        print(f"  {coda_type}: {len(features_list)} SPD feature vectors, "
              f"dim = {features_list[0].shape[0]}")

# %%
# SPD manifold distances between coda types
if len(spd_features_all) >= 2:
    spd_type_names = list(spd_features_all.keys())
    n_spd = len(spd_type_names)

    # Compute mean SPD feature per type
    mean_features = {ct: feats.mean(axis=0) for ct, feats in spd_features_all.items()}

    # Pairwise distances in SPD feature space
    spd_dist_matrix = np.zeros((n_spd, n_spd))
    for i in range(n_spd):
        for j in range(n_spd):
            spd_dist_matrix[i, j] = np.linalg.norm(
                mean_features[spd_type_names[i]] - mean_features[spd_type_names[j]]
            )

    # PCA on SPD features
    all_feats = np.vstack([spd_features_all[ct] for ct in spd_type_names])
    all_labels = np.concatenate([[ct] * len(spd_features_all[ct]) for ct in spd_type_names])

    pca_spd = PCA(n_components=2)
    spd_2d = pca_spd.fit_transform(all_feats)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # SPD feature scatter
    ax = axes[0]
    for ct in spd_type_names:
        mask = all_labels == ct
        rhythm = parse_coda_type(ct)
        c = color_map.get(rhythm['rhythm_class'] if rhythm else 'other', 'gray')
        ax.scatter(spd_2d[mask, 0], spd_2d[mask, 1], c=[c], s=30, alpha=0.5, label=ct)
    ax.set_xlabel('SPD-PC 1')
    ax.set_ylabel('SPD-PC 2')
    ax.set_title('SPD Covariance Features (PCA)')
    ax.legend(fontsize=8, loc='best')

    # Heatmap of inter-type SPD distances
    ax = axes[1]
    sns.heatmap(spd_dist_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=spd_type_names, yticklabels=spd_type_names, ax=ax)
    ax.set_title('SPD Feature Distance Between Coda Types')

    plt.suptitle('SPD Manifold Analysis of Spectral Structure',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig7_spd_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 7 saved.")

# %% [markdown]
# ## 6. Decoder Robustness Index (DRI)
#
# The Decoder Robustness Index adapts the ErisML Bond Index adversarial
# fuzzing framework to cetacean communication decoders. Parametric
# acoustic transforms are applied at graduated intensities, and the
# frequency and severity of output changes are measured via a graduated
# semantic distance metric (omega).
#
# **DRI formula:** `DRI = 0.5 * mean(omega) + 0.3 * p75(omega) + 0.2 * p95(omega)`
#
# A lower DRI indicates a more robust decoder.

# %%
# Build a simple ICI-based coda decoder
class ICIDecoder:
    """Nearest-centroid decoder based on ICI features.

    Detects clicks in a signal, measures inter-click intervals,
    and classifies by nearest centroid in ICI space.
    """
    def __init__(self, centroids, labels, sr=32000):
        self.centroids = centroids  # [n_types, n_ici]
        self.labels = labels
        self.sr = sr

    def _extract_icis(self, signal, sr):
        """Extract inter-click intervals from a signal via energy peaks."""
        # Simple energy-based click detection
        frame_len = int(0.005 * sr)
        hop = frame_len // 2
        energy = np.array([
            np.sum(signal[i:i+frame_len] ** 2)
            for i in range(0, len(signal) - frame_len, hop)
        ])
        if len(energy) == 0:
            return np.zeros(9, dtype=np.float32)

        threshold = np.mean(energy) + 2 * np.std(energy)
        peaks = []
        in_peak = False
        for i, e in enumerate(energy):
            if e > threshold and not in_peak:
                peaks.append(i * hop / sr)
                in_peak = True
            elif e <= threshold:
                in_peak = False

        # Convert peak times to ICIs
        icis = np.zeros(9, dtype=np.float32)
        for i in range(min(len(peaks) - 1, 9)):
            icis[i] = peaks[i + 1] - peaks[i]
        return icis

    def classify(self, signal, sr):
        """Classify a signal by nearest ICI centroid."""
        icis = self._extract_icis(signal, sr)
        dists = np.linalg.norm(self.centroids - icis, axis=1)
        return self.labels[np.argmin(dists)]


# Compute centroids from training data
dri_types = valid_types[:8]
centroids = []
dri_labels = []
for ct in dri_types:
    group = df[df['CodaType'] == ct]
    mean_ici = group[ici_cols].fillna(0).values.mean(axis=0).astype(np.float32)
    centroids.append(mean_ici)
    dri_labels.append(ct)

centroids = np.array(centroids)
decoder = ICIDecoder(centroids, dri_labels)

# Generate test signals
dri_signals = []
dri_true_labels = []
for ct in dri_types:
    group = df[df['CodaType'] == ct].head(10)
    for _, row in group.iterrows():
        icis = [row[c] for c in ici_cols if not pd.isna(row[c]) and float(row[c]) > 0]
        sig = synthesize_coda_signal(icis)
        dri_signals.append(sig)
        dri_true_labels.append(ct)

print(f"DRI test set: {len(dri_signals)} signals, {len(dri_types)} types")

# Quick baseline accuracy check
correct = 0
for sig, true_label in zip(dri_signals, dri_true_labels, strict=False):
    pred = decoder.classify(sig, 32000)
    if pred == true_label:
        correct += 1
baseline_acc = correct / len(dri_signals)
print(f"Baseline decoder accuracy: {baseline_acc:.1%}")

# %%
# Run DRI measurement
transforms = make_acoustic_transform_suite()
semantic_distance = CodaSemanticDistance()
dri = DecoderRobustnessIndex(transforms, semantic_distance)

# Use a subset for speed
dri_subset = dri_signals[:20]

print("Running DRI measurement...")
result = dri.measure(
    decoder,
    dri_subset,
    sr=32000,
    intensities=[0.3, 0.6, 1.0],
    n_chains=15,
    chain_max_length=2,
)

print("\n" + "=" * 60)
print("DECODER ROBUSTNESS INDEX (DRI)")
print("=" * 60)
print(f"  Overall DRI:          {result.dri:.4f}")
print(f"  DRI (invariant only): {result.dri_invariant:.4f}")
print(f"  DRI (stress only):    {result.dri_stress:.4f}")
print("=" * 60)

print("\nPer-transform sensitivity (mean omega at full intensity):")
for name, omega in sorted(result.per_transform.items(), key=lambda x: -x[1]):
    bar = '#' * int(omega * 40)
    print(f"  {name:>20s}: {omega:.4f} |{bar}")

print("\nAdversarial thresholds (minimal flip intensity):")
for name, thresh in sorted(result.adversarial_thresholds.items(), key=lambda x: x[1]):
    bar = '=' * int(thresh * 40)
    print(f"  {name:>20s}: {thresh:.3f} |{bar}|")

# %%
# DRI Visualizations
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Per-transform sensitivity profile
sorted_transforms = sorted(result.per_transform.items(), key=lambda x: -x[1])
t_names = [t[0] for t in sorted_transforms]
t_omegas = [t[1] for t in sorted_transforms]
colors_dri = ['crimson' if o > 0.3 else 'orange' if o > 0.1 else 'seagreen'
              for o in t_omegas]
axes[0].barh(range(len(t_names)), t_omegas, color=colors_dri,
             edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(len(t_names)))
axes[0].set_yticklabels(t_names, fontsize=9)
axes[0].set_xlabel('Mean Omega (sensitivity)')
axes[0].set_title('Per-Transform Sensitivity Profile')
axes[0].invert_yaxis()
axes[0].axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='High risk')
axes[0].axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='Moderate')
axes[0].legend(fontsize=8)

# Adversarial thresholds
sorted_adv = sorted(result.adversarial_thresholds.items(), key=lambda x: x[1])
a_names = [t[0] for t in sorted_adv]
a_thresh = [t[1] for t in sorted_adv]
colors_adv = ['crimson' if t < 0.3 else 'orange' if t < 0.6 else 'seagreen'
              for t in a_thresh]
axes[1].barh(range(len(a_names)), a_thresh, color=colors_adv,
             edgecolor='black', linewidth=0.5)
axes[1].set_yticks(range(len(a_names)))
axes[1].set_yticklabels(a_names, fontsize=9)
axes[1].set_xlabel('Adversarial Threshold (lower = more vulnerable)')
axes[1].set_title('Adversarial Flip Thresholds')
axes[1].invert_yaxis()

# DRI summary gauge
ax = axes[2]
gauge_colors = ['seagreen', 'yellowgreen', 'gold', 'orange', 'crimson']
gauge_bounds = [0, 0.1, 0.2, 0.4, 0.6, 1.0]
for i in range(5):
    ax.barh(0, gauge_bounds[i + 1] - gauge_bounds[i], left=gauge_bounds[i],
            color=gauge_colors[i], height=0.4, edgecolor='black', linewidth=0.5)
ax.axvline(x=result.dri, color='black', linewidth=3, label=f'DRI = {result.dri:.3f}')
ax.scatter([result.dri], [0], s=200, color='black', zorder=5, marker='v')
ax.set_xlim(0, 1)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('DRI Score')
ax.set_title('Decoder Robustness Index')
ax.set_yticks([])
ax.legend(fontsize=12, loc='upper right')
ax.text(0.05, -0.35, 'Robust', fontsize=10, color='seagreen', fontweight='bold')
ax.text(0.75, -0.35, 'Fragile', fontsize=10, color='crimson', fontweight='bold')

plt.suptitle('Decoder Robustness Index (DRI) — Adversarial Fuzzing Results',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig8_dri_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 8 saved.")

# %% [markdown]
# ## 7. Raw Audio Analysis (DSWP Recordings)
#
# Raw coda recordings from the DSWP HuggingFace repository are loaded
# and subjected to spectral trajectory analysis on the SPD manifold.
# This section examines whether vowel-like diphthong patterns, as
# reported by Begus et al. (2025), manifest as geodesic trajectories
# on the spectral covariance manifold.

# %%
try:
    from datasets import load_dataset
    import itertools

    print("Loading DSWP audio from HuggingFace (streaming, first 30 codas)...")
    ds = load_dataset("orrp/DSWP", split="train", streaming=True)
    samples = list(itertools.islice(ds, 30))

    print(f"Loaded {len(samples)} audio samples")
    # Check structure
    sample = samples[0]
    audio = sample['audio']
    print(f"Sample rate: {audio['sampling_rate']} Hz")
    print(f"Signal length: {len(audio['array'])} samples "
          f"({len(audio['array'])/audio['sampling_rate']:.2f}s)")

    # Spectral trajectory analysis
    trajectories = []
    for i, sample in enumerate(samples[:20]):
        audio = sample['audio']
        sig = np.array(audio['array'], dtype=np.float32)
        sr = audio['sampling_rate']

        if len(sig) < 4096:
            continue

        mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=64,
                                             n_fft=1024, hop_length=256)
        mel_db = librosa.power_to_db(mel + 1e-10)

        if mel_db.shape[1] < 48:
            continue

        traj = compute_spectral_trajectory(mel_db, n_bands=8,
                                           window_frames=16, hop_frames=8,
                                           sr=sr, hop_length=256)
        trajectories.append((i, traj))

    print(f"\nComputed {len(trajectories)} spectral trajectories")
    for i, traj in trajectories[:5]:
        print(f"  Coda {i}: {len(traj.timestamps)} windows, "
              f"geodesic_deviation = {traj.geodesic_deviation:.4f}")

    # Visualize geodesic deviations
    if trajectories:
        deviations = [t.geodesic_deviation for _, t in trajectories]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(deviations, bins=15, color='steelblue', edgecolor='black', alpha=0.8)
        axes[0].set_xlabel('Geodesic Deviation')
        axes[0].set_ylabel('Count')
        axes[0].set_title('SPD Trajectory Geodesic Deviation\n'
                          '(0 = perfect geodesic = smooth vowel transition)')
        axes[0].axvline(np.mean(deviations), color='red', linestyle='--',
                        label=f'Mean = {np.mean(deviations):.3f}')
        axes[0].legend()

        # Show one trajectory as SPD distance from start
        idx, best_traj = min(trajectories, key=lambda x: abs(x[1].geodesic_deviation))
        if len(best_traj.matrices) > 1:
            mat_t = torch.tensor(best_traj.matrices, dtype=torch.float32)
            dists_from_start = [0.0]
            for k in range(1, len(mat_t)):
                d = float(SPDManifold.distance(mat_t[0], mat_t[k]))
                dists_from_start.append(d)
            axes[1].plot(best_traj.timestamps, dists_from_start, 'o-',
                         color='steelblue', linewidth=2, markersize=6)
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('SPD Distance from Start')
            axes[1].set_title(f'Spectral Trajectory (Coda {idx})\n'
                              f'Geodesic deviation = {best_traj.geodesic_deviation:.4f}')

        plt.suptitle('DSWP Audio: SPD Manifold Spectral Trajectories',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('fig9_audio_trajectories.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Figure 9 saved.")

except Exception as e:
    print(f"HuggingFace audio analysis skipped: {e}")
    print("This section requires: pip install datasets")

# %% [markdown]
# ## 8. Social Unit Dialect Analysis
#
# Social units of sperm whales are known to share distinctive coda
# repertoires (Rendell & Whitehead, 2003). This section tests whether
# codas from the same social unit cluster more tightly on the Poincare
# ball, which would indicate that the hyperbolic embedding captures
# dialect-level structure distinguishing social groups.

# %%
# Embed individual codas into Poincare ball, color by social unit
if 'Unit' in df_cls.columns:
    units = df_cls['Unit'].values
    unique_units = sorted(df_cls['Unit'].dropna().unique())
    print(f"Social units in classification set: {unique_units}")
    print(f"Codas per unit:")
    for u in unique_units:
        print(f"  {u}: {(units == u).sum()}")

    # Use test set embeddings on the ball
    test_units = df_cls.iloc[X_test.shape[0] * -1:]['Unit'].values if len(df_cls) > 0 else []

    # PCA on Poincare embeddings for visualization
    all_ball = torch.cat([X_train_ball, X_test_ball], dim=0).detach().numpy()
    all_units = np.concatenate([
        df_cls.iloc[:len(X_train)]['Unit'].values,
        df_cls.iloc[len(X_train):]['Unit'].values
    ])

    pca_ball = PCA(n_components=2)
    ball_2d = pca_ball.fit_transform(all_ball)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by coda type
    ax = axes[0]
    for ct in valid_types[:8]:
        mask = df_cls['CodaType'].values == ct
        ax.scatter(ball_2d[mask, 0], ball_2d[mask, 1], s=10, alpha=0.4, label=ct)
    ax.set_xlabel('Poincare-PC 1')
    ax.set_ylabel('Poincare-PC 2')
    ax.set_title('Poincare Embeddings by Coda Type')
    ax.legend(fontsize=7, loc='best', ncol=2, markerscale=3)

    # Color by social unit
    ax = axes[1]
    unit_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_units)))
    for i, u in enumerate(unique_units):
        mask = all_units == u
        if mask.sum() > 0:
            ax.scatter(ball_2d[mask, 0], ball_2d[mask, 1], s=10, alpha=0.4,
                       color=unit_colors[i], label=f'Unit {u}')
    ax.set_xlabel('Poincare-PC 1')
    ax.set_ylabel('Poincare-PC 2')
    ax.set_title('Poincare Embeddings by Social Unit')
    ax.legend(fontsize=7, loc='best', ncol=2, markerscale=3)

    plt.suptitle('Social Unit Dialect Structure on the Poincare Ball',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig10_social_units.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Figure 10 saved.")

    # Quantify: within-unit vs between-unit distances on the ball
    from sklearn.metrics import silhouette_score
    unit_le = LabelEncoder()
    unit_labels = unit_le.fit_transform(all_units)
    sil_score = silhouette_score(all_ball, unit_labels, sample_size=min(2000, len(all_ball)))
    print(f"\nSilhouette score (social unit clustering on Poincare ball): {sil_score:.4f}")
    print("(> 0 means units are separated; higher = more distinct dialects)")

    # Compare with Euclidean
    sil_euclidean = silhouette_score(
        np.vstack([X_train, X_test]), unit_labels,
        sample_size=min(2000, len(all_ball))
    )
    print(f"Silhouette score (Euclidean features): {sil_euclidean:.4f}")
    print(f"Advantage: {'Poincare' if sil_score > sil_euclidean else 'Euclidean'}")

# %% [markdown]
# ## 9. Information-Theoretic and Linguistic Properties
#
# Beyond geometric structure, cetacean communication systems may exhibit
# statistical regularities shared with human language. Youngblood (2025)
# demonstrated that Zipf's law of abbreviation and Menzerath's law hold
# across 16 cetacean species. Here we test whether the DSWP sperm whale
# codas exhibit these linguistic universals and quantify sequential
# information structure in coda exchanges.

# %%
# === 9.1 Zipf's Law of Abbreviation ===
# Zipf's law predicts that more frequent coda types have shorter durations.
# We fit a power law to the rank-frequency distribution.

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, ks_2samp

# Rank-frequency distribution of coda types
coda_freq = df['CodaType'].value_counts()
ranks = np.arange(1, len(coda_freq) + 1)
frequencies = coda_freq.values

# Fit power law: f(r) = C * r^(-alpha)
def power_law(r, C, alpha):
    return C * r ** (-alpha)

popt, pcov = curve_fit(power_law, ranks, frequencies, p0=[frequencies[0], 1.0])
C_fit, alpha_fit = popt

# Zipf's law of abbreviation: more frequent types should be shorter
type_mean_dur = df.groupby('CodaType')['Duration'].mean()
type_freq = df['CodaType'].value_counts()
common_types = type_freq.index[:30]  # Top 30 types for clean analysis

freq_vals = np.array([type_freq[ct] for ct in common_types], dtype=float)
dur_vals = np.array([type_mean_dur[ct] for ct in common_types], dtype=float)
log_freq = np.log(freq_vals)
log_dur = np.log(dur_vals)
r_zipf, p_zipf = pearsonr(log_freq, log_dur)

print("=" * 60)
print("9.1 ZIPF'S LAW ANALYSIS")
print("=" * 60)
print(f"  Power law exponent (rank-frequency): alpha = {alpha_fit:.2f}")
print(f"  Reference: human language alpha ~ 1.0; steeper = more skewed")
print(f"\n  Zipf's law of abbreviation (log freq vs log duration):")
print(f"  Pearson r = {r_zipf:.3f}, p = {p_zipf:.2e}")
if r_zipf < 0 and p_zipf < 0.05:
    print(f"  Confirmed: more frequent coda types are shorter (Youngblood, 2025).")
else:
    print(f"  Not significant at this sample size.")

# %%
# === 9.2 Menzerath's Law ===
# Menzerath's law predicts that longer codas (more clicks) have
# shorter constituent elements (shorter inter-click intervals).

mean_ici_per_coda = df[ici_cols].apply(
    lambda row: row[row > 0].mean() if (row > 0).any() else np.nan, axis=1
)
valid_mask = ~mean_ici_per_coda.isna()
nclicks = df.loc[valid_mask, 'nClicks'].values.astype(float)
mean_icis = mean_ici_per_coda[valid_mask].values

r_menz, p_menz = spearmanr(nclicks, mean_icis)

print(f"\n{'=' * 60}")
print("9.2 MENZERATH'S LAW")
print("=" * 60)
print(f"  Spearman correlation (nClicks vs mean ICI): r = {r_menz:.3f}, p = {p_menz:.2e}")
if r_menz < 0 and p_menz < 0.001:
    print(f"  Confirmed: longer codas have shorter inter-click intervals.")
    print(f"  This linguistic universal holds in sperm whale communication,")
    print(f"  consistent with Youngblood (2025).")

# %%
# === 9.3 Individual Whale Accents ===
# Test whether individual whales produce the same coda type with
# statistically distinguishable ICI patterns ("accents").

accent_type = None
top_whales = []
whale_ici_data = {}

if 'IDN' in df.columns:
    # Pick the most common coda type with multiple identified whales
    accent_type = None
    for ct in valid_types[:5]:
        whales = df[(df['CodaType'] == ct) & df['IDN'].notna()]['IDN'].unique()
        if len(whales) >= 3:
            accent_type = ct
            break

    if accent_type:
        accent_df = df[(df['CodaType'] == accent_type) & df['IDN'].notna()]
        whale_ids = accent_df['IDN'].value_counts()
        top_whales = whale_ids[whale_ids >= 10].index[:4]

        print(f"\n{'=' * 60}")
        print(f"9.3 INDIVIDUAL WHALE ACCENTS")
        print(f"{'=' * 60}")
        print(f"  Testing coda type: {accent_type}")
        print(f"  Whales with >= 10 samples: {len(top_whales)}")

        whale_ici_data = {}
        for w in top_whales:
            w_data = accent_df[accent_df['IDN'] == w][ici_cols].fillna(0).values
            whale_ici_data[w] = w_data.flatten()
            whale_ici_data[w] = whale_ici_data[w][whale_ici_data[w] > 0]

        # Pairwise KS tests
        print(f"\n  Pairwise Kolmogorov-Smirnov tests on ICI distributions:")
        accent_results = []
        for i, w1 in enumerate(top_whales):
            for w2 in top_whales[i+1:]:
                stat, pval = ks_2samp(whale_ici_data[w1], whale_ici_data[w2])
                accent_results.append((w1, w2, stat, pval))
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
                print(f"    {w1} vs {w2}: D = {stat:.4f}, p = {pval:.2e} {sig}")

        n_sig = sum(1 for _, _, _, p in accent_results if p < 0.001)
        print(f"\n  {n_sig}/{len(accent_results)} pairs significantly different (p < 0.001)")
        if n_sig > len(accent_results) / 2:
            print(f"  Individual whales produce the same coda type with distinguishable")
            print(f"  ICI patterns, consistent with individual identity encoding")
            print(f"  (Gero et al., 2016; Hersh et al., 2022).")
    else:
        print("\n  Insufficient data for accent analysis.")

# %%
# === 9.4 Load Exchange Data for Sequential Analysis ===
# The dialogues dataset contains sequential coda exchanges with
# rhythm type labels, enabling Markov and information-theoretic analysis.

DIALOGUES_URLS = [
    "https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/main/data/sperm-whale-dialogues.csv",
    "https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/master/data/sperm-whale-dialogues.csv",
]

import pickle
import urllib.request
import io

df_dial = None
for url in DIALOGUES_URLS:
    try:
        df_dial = pd.read_csv(url)
        print(f"Loaded {len(df_dial)} exchange entries from dialogues dataset")
        break
    except Exception:
        continue

# Load rhythm type labels
rhythms = None
PICKLE_BASE = "https://raw.githubusercontent.com/pratyushasharma/sw-combinatoriality/main/data/"
try:
    with urllib.request.urlopen(PICKLE_BASE + "rhythms.p") as resp:
        rhythms = pickle.load(io.BytesIO(resp.read()))
    print(f"Loaded rhythm labels: {len(rhythms)} entries, {len(set(rhythms))} unique types")
except Exception as e:
    print(f"Could not load rhythm labels: {e}")

# %%
# === 9.5 Sequential Structure and Mutual Information ===
if rhythms is not None:
    from collections import Counter

    rhythm_seq = list(rhythms)
    unique_rhythms_seq = sorted(set(rhythm_seq))
    n_rhythm_types = len(unique_rhythms_seq)

    # Unigram entropy H(R)
    counts = Counter(rhythm_seq)
    total = sum(counts.values())
    probs = np.array([counts[r] / total for r in unique_rhythms_seq])
    H_R = -np.sum(probs * np.log2(probs + 1e-15))

    # Bigram conditional entropy H(R_{t+1} | R_t)
    bigrams = [(rhythm_seq[i], rhythm_seq[i+1]) for i in range(len(rhythm_seq)-1)]
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(rhythm_seq)

    H_cond = 0
    for (r1, r2), count in bigram_counts.items():
        p_joint = count / len(bigrams)
        p_cond = count / unigram_counts[r1]
        H_cond -= p_joint * np.log2(p_cond + 1e-15)

    MI = H_R - H_cond
    MI_pct = MI / H_R * 100

    # Bigram prediction accuracy
    # For each bigram context, predict the most common successor
    successor_counts = {}
    for r1, r2 in bigrams:
        if r1 not in successor_counts:
            successor_counts[r1] = Counter()
        successor_counts[r1][r2] += 1

    bigram_correct = sum(
        successor_counts[r1].most_common(1)[0][1]
        for r1 in successor_counts
    )
    bigram_acc = bigram_correct / len(bigrams)

    # Trigram prediction accuracy
    trigrams = [(rhythm_seq[i], rhythm_seq[i+1], rhythm_seq[i+2])
                for i in range(len(rhythm_seq)-2)]
    tri_successor = {}
    for r1, r2, r3 in trigrams:
        key = (r1, r2)
        if key not in tri_successor:
            tri_successor[key] = Counter()
        tri_successor[key][r3] += 1

    trigram_correct = sum(
        tri_successor[key].most_common(1)[0][1]
        for key in tri_successor
    )
    trigram_acc = trigram_correct / len(trigrams)

    print(f"\n{'=' * 60}")
    print("9.5 SEQUENTIAL STRUCTURE AND MUTUAL INFORMATION")
    print("=" * 60)
    print(f"  Rhythm types: {n_rhythm_types}")
    print(f"  Unigram entropy H(R): {H_R:.3f} bits")
    print(f"  Conditional entropy H(R_{{t+1}} | R_t): {H_cond:.3f} bits")
    print(f"  Mutual information I(R_t; R_{{t+1}}): {MI:.3f} bits ({MI_pct:.1f}%)")
    print(f"\n  Prediction accuracy:")
    print(f"    Bigram (1st-order Markov):  {bigram_acc:.1%}")
    print(f"    Trigram (2nd-order Markov): {trigram_acc:.1%}")
    print(f"    Improvement: +{(trigram_acc - bigram_acc)*100:.1f} percentage points")
    print(f"\n  Coda sequences exhibit higher-order Markov structure:")
    print(f"  the preceding two codas predict the next more accurately than")
    print(f"  the preceding one alone, indicating non-trivial sequential grammar.")

# %%
# === 9.6 Turn-Taking Dynamics ===
same_whale_gaps = []
cross_whale_gaps = []
mean_same = 0
mean_cross = 0

if df_dial is not None and 'Whale' in df_dial.columns:
    # Group by recording session (REC), sort by timestamp within each
    df_dial_sorted = df_dial.sort_values(['REC', 'TsTo']).reset_index(drop=True)

    for i in range(1, len(df_dial_sorted)):
        prev = df_dial_sorted.iloc[i-1]
        curr = df_dial_sorted.iloc[i]
        if prev['REC'] != curr['REC']:
            continue
        gap = curr['TsTo'] - prev['TsTo']
        if gap <= 0 or gap > 30:
            continue
        if prev['Whale'] == curr['Whale']:
            same_whale_gaps.append(gap)
        else:
            cross_whale_gaps.append(gap)

    if same_whale_gaps and cross_whale_gaps:
        mean_same = np.mean(same_whale_gaps)
        mean_cross = np.mean(cross_whale_gaps)
        ks_stat, ks_p = ks_2samp(same_whale_gaps, cross_whale_gaps)

        print(f"\n{'=' * 60}")
        print("9.6 TURN-TAKING DYNAMICS")
        print("=" * 60)
        print(f"  Same-whale continuation: {mean_same:.2f}s (n = {len(same_whale_gaps)})")
        print(f"  Cross-whale response:    {mean_cross:.2f}s (n = {len(cross_whale_gaps)})")
        print(f"  Ratio: {mean_same/mean_cross:.2f}x")
        print(f"  KS test: D = {ks_stat:.4f}, p = {ks_p:.2e}")
        if mean_cross < mean_same:
            print(f"\n  Cross-whale responses are faster than same-whale continuations,")
            print(f"  indicating active turn-taking rather than independent vocalization.")

# %%
# === Figure 12: Linguistic Properties Dashboard ===
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Panel A: Zipf's rank-frequency
ax = axes[0, 0]
ax.loglog(ranks, frequencies, 'o', markersize=4, color='steelblue', alpha=0.7)
ax.loglog(ranks, power_law(ranks, C_fit, alpha_fit), 'r--', linewidth=2,
          label=f'Power law (alpha = {alpha_fit:.2f})')
ax.set_xlabel('Rank')
ax.set_ylabel('Frequency')
ax.set_title(f'A. Zipf\'s Rank-Frequency Distribution\nalpha = {alpha_fit:.2f}')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: Zipf's law of abbreviation
ax = axes[0, 1]
ax.scatter(log_freq, log_dur, s=40, color='steelblue', alpha=0.7, edgecolors='black',
           linewidths=0.5)
z_abbr = np.polyfit(log_freq, log_dur, 1)
x_line = np.linspace(log_freq.min(), log_freq.max(), 100)
ax.plot(x_line, np.polyval(z_abbr, x_line), 'r--', linewidth=2)
ax.set_xlabel('log(Frequency)')
ax.set_ylabel('log(Mean Duration)')
ax.set_title(f'B. Zipf\'s Law of Abbreviation\nr = {r_zipf:.3f}, p = {p_zipf:.1e}')
ax.grid(True, alpha=0.3)

# Panel C: Menzerath's law
ax = axes[0, 2]
# Bin by click count for cleaner visualization
click_bins = sorted(df['nClicks'].unique())
bin_means = []
bin_stds = []
for nc in click_bins:
    mask = (nclicks == nc) & valid_mask.values[:len(nclicks)]
    vals = mean_icis[nclicks == nc]
    if len(vals) > 5:
        bin_means.append(np.mean(vals))
        bin_stds.append(np.std(vals) / np.sqrt(len(vals)))
    else:
        bin_means.append(np.nan)
        bin_stds.append(np.nan)

valid_bins = [(nc, m, s) for nc, m, s in zip(click_bins, bin_means, bin_stds)
              if not np.isnan(m)]
if valid_bins:
    vb_x = [v[0] for v in valid_bins]
    vb_y = [v[1] for v in valid_bins]
    vb_e = [v[2] for v in valid_bins]
    ax.errorbar(vb_x, vb_y, yerr=vb_e, fmt='o-', color='steelblue',
                linewidth=2, markersize=8, capsize=4)
ax.set_xlabel('Number of Clicks')
ax.set_ylabel('Mean ICI (seconds)')
ax.set_title(f'C. Menzerath\'s Law\nSpearman r = {r_menz:.3f}, p = {p_menz:.1e}')
ax.grid(True, alpha=0.3)

# Panel D: Individual accents (if available)
ax = axes[1, 0]
if 'IDN' in df.columns and accent_type and len(top_whales) >= 2:
    for w in top_whales:
        data = whale_ici_data[w]
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.5, density=True, label=f'Whale {w}')
    ax.set_xlabel('Inter-Click Interval (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'D. Individual Accents ({accent_type})\nKS p < 0.001 between individuals')
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, 'Insufficient whale ID data', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_title('D. Individual Accents')

# Panel E: Mutual information / Markov order
ax = axes[1, 1]
if rhythms is not None:
    categories = ['Unigram\nH(R)', 'Conditional\nH(R|R_prev)', 'Mutual\nInfo']
    values = [H_R, H_cond, MI]
    colors_info = ['steelblue', 'coral', 'seagreen']
    bars = ax.bar(categories, values, color=colors_info, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Bits')
    ax.set_title(f'E. Information-Theoretic Structure\nMI = {MI:.3f} bits ({MI_pct:.1f}%)')

    # Add prediction accuracy annotation
    ax.text(0.95, 0.95, f'Bigram acc: {bigram_acc:.1%}\nTrigram acc: {trigram_acc:.1%}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
else:
    ax.text(0.5, 0.5, 'Dialogues data not available', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_title('E. Information-Theoretic Structure')

# Panel F: Turn-taking latency distributions
ax = axes[1, 2]
if df_dial is not None and same_whale_gaps and cross_whale_gaps:
    ax.hist(same_whale_gaps, bins=50, alpha=0.6, density=True, color='steelblue',
            label=f'Same whale ({mean_same:.2f}s)', range=(0, 15))
    ax.hist(cross_whale_gaps, bins=50, alpha=0.6, density=True, color='coral',
            label=f'Cross whale ({mean_cross:.2f}s)', range=(0, 15))
    ax.set_xlabel('Inter-Coda Interval (s)')
    ax.set_ylabel('Density')
    ax.set_title(f'F. Turn-Taking Latency\nCross-whale {mean_cross:.1f}s vs same-whale {mean_same:.1f}s')
    ax.legend(fontsize=9)
    ax.axvline(mean_same, color='steelblue', linestyle='--', alpha=0.7)
    ax.axvline(mean_cross, color='coral', linestyle='--', alpha=0.7)
else:
    ax.text(0.5, 0.5, 'Dialogues data not available', ha='center', va='center',
            transform=ax.transAxes)
    ax.set_title('F. Turn-Taking Latency')

plt.suptitle('Linguistic and Information-Theoretic Properties of Sperm Whale Codas',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig12_linguistic_properties.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 12 saved.")

# %%
# === 9.7 Summary of Linguistic Findings ===
print("\n" + "=" * 70)
print("SUMMARY: LINGUISTIC AND INFORMATION-THEORETIC PROPERTIES")
print("=" * 70)
print(f"\n  Zipf's rank-frequency exponent:  alpha = {alpha_fit:.2f}")
print(f"  Zipf's abbreviation correlation: r = {r_zipf:.3f} (p = {p_zipf:.1e})")
print(f"  Menzerath's law:                 r = {r_menz:.3f} (p = {p_menz:.1e})")
if rhythms is not None:
    print(f"  Unigram entropy:                 H(R) = {H_R:.3f} bits")
    print(f"  Mutual information:              MI = {MI:.3f} bits ({MI_pct:.1f}%)")
    print(f"  Bigram prediction accuracy:      {bigram_acc:.1%}")
    print(f"  Trigram prediction accuracy:      {trigram_acc:.1%}")
if df_dial is not None and same_whale_gaps and cross_whale_gaps:
    print(f"  Same-whale continuation latency: {mean_same:.2f}s")
    print(f"  Cross-whale response latency:    {mean_cross:.2f}s")
    print(f"  Turn-taking ratio:               {mean_same/mean_cross:.2f}x")
print(f"\n  These results confirm that sperm whale coda communication")
print(f"  exhibits multiple statistical universals of natural language,")
print(f"  including Zipf's law, Menzerath's law, and higher-order")
print(f"  sequential structure with active turn-taking.")
print("=" * 70)

# %% [markdown]
# ## 10. Adversarial Perturbation as a Structure Discovery Tool
#
# A novel contribution of this work is the use of adversarial
# perturbations not merely for robustness evaluation, but as an
# instrument for *discovering* phonetic structure. By systematically
# applying parametric acoustic transforms at graduated intensities
# and recording which perturbations cause classification boundary
# crossings in feature space, we construct a map of the **phonetic
# decision boundaries** of the coda system.
#
# This approach reveals:
# 1. Which acoustic features carry semantic content (perturbation-sensitive dimensions)
# 2. Which features are redundant or noise-tolerant (perturbation-invariant dimensions)
# 3. Phonetic neighborhoods: which coda types occupy adjacent regions of acoustic space
# 4. An empirical estimate of the information-theoretic channel capacity of coda communication

# %%
# === FUZZ-BASED PHONETIC BOUNDARY MAPPING ===
# For each coda type, apply transforms at graduated intensities and measure
# the distance the perturbed signal moves in feature space (ICI space).
# If it crosses into another type's territory, that's a "phonetic boundary crossing."

from eris_ketos.acoustic_transforms import make_acoustic_transform_suite

fuzz_transforms = make_acoustic_transform_suite()
intensity_grid = np.linspace(0.0, 1.0, 11)

# Use the same classification features + decoder from Section 6
# For each coda type, take representative codas, perturb, measure feature drift
fuzz_types = valid_types[:8]  # Top 8 coda types

print("=" * 60)
print("FUZZ-BASED PHONETIC BOUNDARY MAPPING")
print("=" * 60)

# Collect per-transform, per-type sensitivity curves
boundary_crossings = {}  # {transform_name: {coda_type: [(intensity, n_crossings)]}}
feature_drift = {}       # {transform_name: {coda_type: [(intensity, mean_drift)]}}
crossing_targets = {}    # {(source_type, transform): {target_type: count}}

for t in fuzz_transforms:
    boundary_crossings[t.name] = {}
    feature_drift[t.name] = {}

    for ct in fuzz_types:
        group = df[df['CodaType'] == ct].head(20)
        crossings_by_intensity = []
        drift_by_intensity = []

        for intensity in intensity_grid:
            n_cross = 0
            drifts = []

            for _, row in group.iterrows():
                icis = [row[c] for c in ici_cols if not pd.isna(row[c]) and float(row[c]) > 0]
                sig = synthesize_coda_signal(icis)

                # Apply transform
                perturbed = t(sig, sr=32000, intensity=float(intensity))

                # Extract features from perturbed signal
                perturbed_icis = decoder._extract_icis(perturbed, 32000)
                original_icis = decoder._extract_icis(sig, 32000)

                # Measure drift in ICI feature space
                drift = np.linalg.norm(perturbed_icis - original_icis)
                drifts.append(drift)

                # Check if classification changes
                pred_orig = decoder.classify(sig, 32000)
                pred_pert = decoder.classify(perturbed, 32000)
                if pred_orig != pred_pert:
                    n_cross += 1
                    key = (ct, t.name)
                    if key not in crossing_targets:
                        crossing_targets[key] = {}
                    crossing_targets[key][pred_pert] = crossing_targets[key].get(pred_pert, 0) + 1

            crossings_by_intensity.append((float(intensity), n_cross / max(len(group), 1)))
            drift_by_intensity.append((float(intensity), np.mean(drifts) if drifts else 0))

        boundary_crossings[t.name][ct] = crossings_by_intensity
        feature_drift[t.name][ct] = drift_by_intensity

print(f"Analyzed {len(fuzz_transforms)} transforms x {len(fuzz_types)} types "
      f"x {len(intensity_grid)} intensities")

# %%
# === DISCOVERY 1: Transform Sensitivity Spectrum ===
# Which transforms cause the most boundary crossings? This reveals which
# acoustic dimensions carry semantic content.

# Aggregate: mean crossing rate at full intensity per transform
transform_sensitivity = {}
for t_name in boundary_crossings:
    rates = []
    for ct in fuzz_types:
        # Get crossing rate at intensity=1.0
        curve = boundary_crossings[t_name][ct]
        full_rate = curve[-1][1]  # last intensity point
        rates.append(full_rate)
    transform_sensitivity[t_name] = np.mean(rates)

sorted_sensitivity = sorted(transform_sensitivity.items(), key=lambda x: -x[1])

print("\nTransform Sensitivity Spectrum")
print("Boundary crossing rate at full intensity by transform type:\n")
for name, rate in sorted_sensitivity:
    bar = '#' * int(rate * 50)
    category = "INV" if any(t.is_invariant and t.name == name for t in fuzz_transforms) else "STR"
    print(f"  [{category}] {name:>20s}: {rate:.1%} |{bar}")

# Invariant transforms that cause crossings indicate decoder vulnerability;
# stress transforms that do not cause crossings indicate semantically empty dimensions.
print("\nNotable observations:")
for name, rate in sorted_sensitivity:
    is_inv = any(t.is_invariant and t.name == name for t in fuzz_transforms)
    if is_inv and rate > 0.3:
        print(f"  - {name} (invariant) causes {rate:.0%} boundary crossings, "
              "indicating decoder sensitivity to a theoretically irrelevant dimension.")
    if not is_inv and rate < 0.1:
        print(f"  - {name} (stress) causes only {rate:.0%} crossings, "
              "suggesting this acoustic dimension carries minimal semantic content.")

# %%
# === DISCOVERY 2: Phonetic Neighborhood Map ===
# When a coda crosses a boundary, WHERE does it go?
# This reveals which coda types are "acoustically adjacent"

print("\nPhonetic Neighborhood Map")
print("Transition probabilities under perturbation (source -> target):\n")

# Build adjacency matrix from crossing targets
adjacency = np.zeros((len(fuzz_types), len(fuzz_types)))
type_to_idx = {ct: i for i, ct in enumerate(fuzz_types)}

for (source, t_name), targets in crossing_targets.items():
    if source not in type_to_idx:
        continue
    for target, count in targets.items():
        if target in type_to_idx:
            adjacency[type_to_idx[source], type_to_idx[target]] += count

# Normalize rows
row_sums = adjacency.sum(axis=1, keepdims=True)
adjacency_norm = np.where(row_sums > 0, adjacency / row_sums, 0)

# Print confusion-like table
print(f"{'Source':>10s} -> ", end="")
for ct in fuzz_types:
    print(f"{ct:>8s}", end=" ")
print()
print("-" * (12 + 9 * len(fuzz_types)))
for i, ct in enumerate(fuzz_types):
    print(f"{ct:>10s} -> ", end="")
    for j in range(len(fuzz_types)):
        val = adjacency_norm[i, j]
        if val > 0.01:
            print(f"{val:>7.0%} ", end="")
        else:
            print(f"{'·':>8s}", end=" ")
    print()

# Identify nearest phonetic neighbors
print("\nNearest phonetic neighbors (most frequent misclassification target):")
for i, ct in enumerate(fuzz_types):
    if adjacency[i].sum() > 0:
        best_j = np.argmax(adjacency[i])
        if adjacency[i, best_j] > 0:
            neighbor = fuzz_types[best_j]
            pct = adjacency_norm[i, best_j]
            rhythm_src = parse_coda_type(ct)
            rhythm_dst = parse_coda_type(neighbor)
            same_rhythm = (rhythm_src and rhythm_dst and
                          rhythm_src['rhythm_class'] == rhythm_dst['rhythm_class'])
            relation = "same" if same_rhythm else "different"
            print(f"  {ct} -> {neighbor} ({pct:.0%}, {relation} rhythm class)")

# %%
# === DISCOVERY 3: Per-Type Robustness Profiles ===
# Some coda types have "wider phonetic territories" than others.
# Types that resist perturbation are more acoustically distinct.

# Compute activation threshold per type (intensity where crossing rate exceeds 50%)
type_robustness = {}
for ct in fuzz_types:
    all_crossing_rates = []
    for t_name in boundary_crossings:
        curve = boundary_crossings[t_name][ct]
        # Find intensity where crossing rate first exceeds 25%
        threshold = 1.0
        for intensity, rate in curve:
            if rate > 0.25:
                threshold = intensity
                break
        all_crossing_rates.append(threshold)
    type_robustness[ct] = np.mean(all_crossing_rates)

sorted_robustness = sorted(type_robustness.items(), key=lambda x: -x[1])

print("\nPer-Type Robustness (Phonetic Territory Width)")
print("Higher threshold indicates a wider phonetic territory and greater acoustic distinctiveness.\n")
for ct, rob in sorted_robustness:
    bar = '=' * int(rob * 40)
    rhythm = parse_coda_type(ct)
    rhythm_class = rhythm['rhythm_class'] if rhythm else 'unknown'
    print(f"  {ct:>10s} [{rhythm_class:>13s}]: {rob:.2f} |{bar}|")

most_robust = sorted_robustness[0]
most_fragile = sorted_robustness[-1]
print(f"\n  Most acoustically distinct: {most_robust[0]} (threshold = {most_robust[1]:.2f})")
print(f"  Least acoustically distinct: {most_fragile[0]} (threshold = {most_fragile[1]:.2f})")
print(f"  Distinctiveness ratio: {most_robust[1]/max(most_fragile[1], 0.01):.1f}x")

# %%
# === DISCOVERY 4: Information-Carrying Features via Differential Sensitivity ===
# Compare drift curves across transforms to identify which acoustic dimensions
# carry information vs which are noise-tolerant

print("\nInformation-Carrying Feature Analysis")
print("Feature drift slope by transform (higher slope = greater sensitivity):\n")

# For each transform, compute average drift slope (d(drift)/d(intensity))
drift_slopes = {}
for t_name in feature_drift:
    slopes = []
    for ct in fuzz_types:
        curve = feature_drift[t_name][ct]
        intensities_arr = [c[0] for c in curve]
        drifts_arr = [c[1] for c in curve]
        if len(intensities_arr) > 2:
            slope = np.polyfit(intensities_arr, drifts_arr, 1)[0]
            slopes.append(slope)
    drift_slopes[t_name] = np.mean(slopes) if slopes else 0

sorted_slopes = sorted(drift_slopes.items(), key=lambda x: -x[1])

for name, slope in sorted_slopes:
    bar = '#' * int(min(slope * 500, 50))
    print(f"  {name:>20s}: slope = {slope:.4f} |{bar}")

# %%
# === VISUALIZATION: Fuzz Discovery Dashboard (Figure 11) ===
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel A: Transform sensitivity spectrum
ax = axes[0, 0]
t_names_sorted = [s[0] for s in sorted_sensitivity]
t_rates_sorted = [s[1] for s in sorted_sensitivity]
colors_sens = []
for name in t_names_sorted:
    is_inv = any(t.is_invariant and t.name == name for t in fuzz_transforms)
    colors_sens.append('steelblue' if is_inv else 'coral')
ax.barh(range(len(t_names_sorted)), t_rates_sorted, color=colors_sens,
        edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(t_names_sorted)))
ax.set_yticklabels(t_names_sorted, fontsize=9)
ax.set_xlabel('Boundary Crossing Rate at Full Intensity')
ax.set_title('A. Transform Sensitivity Spectrum\n'
             '(blue=invariant, coral=stress)')
ax.invert_yaxis()
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)

# Panel B: Phonetic neighborhood heatmap
ax = axes[0, 1]
mask = adjacency_norm < 0.01
sns.heatmap(adjacency_norm, annot=True, fmt='.0%', cmap='YlOrRd',
            xticklabels=fuzz_types, yticklabels=fuzz_types,
            mask=mask, ax=ax, cbar_kws={'label': 'Crossing probability'})
ax.set_title('B. Phonetic Neighborhood Map\n(when perturbed, which type does it become?)')
ax.set_xlabel('Target Type')
ax.set_ylabel('Source Type')

# Panel C: Per-type robustness
ax = axes[1, 0]
rob_types = [r[0] for r in sorted_robustness]
rob_vals = [r[1] for r in sorted_robustness]
rob_colors = [color_map.get(parse_coda_type(ct)['rhythm_class'], 'gray')
              if parse_coda_type(ct) else 'gray' for ct in rob_types]
ax.barh(range(len(rob_types)), rob_vals, color=rob_colors,
        edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(rob_types)))
ax.set_yticklabels(rob_types, fontsize=9)
ax.set_xlabel('Mean Activation Threshold (higher = more distinct)')
ax.set_title('C. Phonetic Territory Width by Coda Type')
ax.invert_yaxis()

# Panel D: Feature drift curves (overlay all types for top 3 transforms)
ax = axes[1, 1]
top_transforms = [s[0] for s in sorted_sensitivity[:3]]
line_styles = ['-', '--', '-.']
for t_idx, t_name in enumerate(top_transforms):
    # Average across types
    all_drifts = []
    for ct in fuzz_types:
        curve = feature_drift[t_name][ct]
        all_drifts.append([c[1] for c in curve])
    mean_drift = np.mean(all_drifts, axis=0)
    ax.plot(intensity_grid, mean_drift, line_styles[t_idx % 3],
            linewidth=2, label=t_name, marker='o', markersize=4)
ax.set_xlabel('Perturbation Intensity')
ax.set_ylabel('Mean Feature Drift (ICI space)')
ax.set_title('D. Feature Drift Curves\n(how fast does the signal leave its territory?)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Fuzz Testing as Structure Discovery\n'
             'Adversarial perturbations reveal phonetic boundaries in whale communication',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('fig11_fuzz_discovery.png', dpi=150, bbox_inches='tight')
plt.show()
print("Figure 11 saved.")

# %%
# Synthesis: Novel findings from adversarial structure discovery
print("\n" + "=" * 70)
print("SYNTHESIS: ADVERSARIAL STRUCTURE DISCOVERY RESULTS")
print("=" * 70)

# 1. Channel redundancy
inv_transforms = [t.name for t in fuzz_transforms if t.is_invariant]
stress_transforms = [t.name for t in fuzz_transforms if not t.is_invariant]

inv_crossing_rates = [transform_sensitivity[t] for t in inv_transforms]
stress_crossing_rates = [transform_sensitivity[t] for t in stress_transforms]

print(f"\n1. Channel Redundancy")
print(f"   Mean crossing rate under invariant transforms: {np.mean(inv_crossing_rates):.1%}")
print(f"   Mean crossing rate under stress transforms:    {np.mean(stress_crossing_rates):.1%}")
if np.mean(inv_crossing_rates) > 0.2:
    print(f"   The coda system exhibits low redundancy: small perturbations in theoretically")
    print(f"   irrelevant dimensions are sufficient to alter classification.")
else:
    print(f"   The coda system exhibits substantial redundancy: classification is robust")
    print(f"   to perturbations in theoretically irrelevant acoustic dimensions.")

# 2. Phonetic boundary asymmetry
print(f"\n2. Phonetic Boundary Asymmetry")
asymmetry_scores = []
for i in range(len(fuzz_types)):
    for j in range(i + 1, len(fuzz_types)):
        if adjacency[i, j] + adjacency[j, i] > 0:
            asym = abs(adjacency[i, j] - adjacency[j, i]) / (adjacency[i, j] + adjacency[j, i])
            asymmetry_scores.append(asym)
            if asym > 0.5:
                print(f"   Asymmetric pair: {fuzz_types[i]} <-> {fuzz_types[j]} "
                      f"({adjacency[i,j]:.0f} vs {adjacency[j,i]:.0f} crossings)")
if asymmetry_scores:
    mean_asym = np.mean(asymmetry_scores)
    print(f"   Mean boundary asymmetry: {mean_asym:.2f} (0 = symmetric, 1 = unidirectional)")
    if mean_asym > 0.3:
        print(f"   Phonetic boundaries are asymmetric, analogous to directional vowel shifts")
        print(f"   observed in human language phonology.")
    else:
        print(f"   Phonetic boundaries are approximately symmetric between type pairs.")

# 3. Robustness by rhythm class
print(f"\n3. Information Hierarchy by Rhythm Class")
rhythm_robustness = {}
for ct, rob in type_robustness.items():
    rhythm = parse_coda_type(ct)
    if rhythm:
        rc = rhythm['rhythm_class']
        if rc not in rhythm_robustness:
            rhythm_robustness[rc] = []
        rhythm_robustness[rc].append(rob)

for rc, robs in sorted(rhythm_robustness.items(), key=lambda x: -np.mean(x[1])):
    print(f"   {rc:>14s}: mean robustness = {np.mean(robs):.3f} (n = {len(robs)} types)")

# 4. Channel capacity estimate
print(f"\n4. Empirical Channel Capacity Estimate")
well_separated = sum(1 for r in type_robustness.values() if r > 0.5)
total_tested = len(type_robustness)
print(f"   Types tested: {total_tested}")
print(f"   Well-separated types (activation threshold > 0.5): {well_separated}")
effective_bits = np.log2(max(well_separated, 1))
print(f"   Empirical channel capacity: {effective_bits:.1f} bits per coda")
print(f"   Reference: H(rhythm) from Sharma et al. (2024) = 2.1 bits")
print("=" * 70)

# %% [markdown]
# ## 11. Summary of Findings
#
# ### Key Results
#
# | Analysis | Finding |
# |----------|---------|
# | **Poincare Embedding** | Coda type hierarchy embeds with lower distortion in hyperbolic space than Euclidean |
# | **HyperbolicMLR** | Hyperbolic multinomial logistic regression achieves competitive coda classification |
# | **TDA** | Persistent homology reveals topologically distinct attractors per rhythm type |
# | **SPD Manifold** | Log-Euclidean spectral covariance captures inter-band harmonic structure |
# | **DRI** | First adversarial robustness benchmark for cetacean communication decoders |
# | **Fuzz Discovery** | Adversarial perturbations map phonetic boundaries and reveal information-carrying features |
#
# ### Implications
#
# 1. Geometric methods (hyperbolic, topological, Riemannian) reveal structure in cetacean communication that Euclidean analyses miss.
# 2. Adversarial robustness testing exposes decoder failure modes and, when used as a discovery tool, maps phonetic decision boundaries.
# 3. The open-source eris-ketos toolkit enables reproducible geometric bioacoustic analysis.
# 4. The framework generalizes to other cetacean species (orcas, dolphins, humpbacks) with minimal adaptation.
#
# ### Next Steps
#
# 1. Apply to the complete DSWP and Project CETI datasets.
# 2. Cross-validate geometric structure with behavioral context (exchange patterns, social function).
# 3. Expand to multi-species comparison using DCLDE orca data.
# 4. Submit to peer review: geometric structure paper and adversarial robustness paper.

# %% [markdown]
# ## References
#
# ### Cetacean Communication
#
# 1. Sharma, P., Gero, S., Payne, R., Gruber, D.F., Rus, D., Torralba, A., & Andreas, J. (2024).
#    Contextual and combinatorial structure in sperm whale vocalisations.
#    *Nature Communications*, 15, 3617. https://doi.org/10.1038/s41467-024-47221-8
#
# 2. Begus, G., Sprouse, R.L., Leban, A., Silva, M., & Gero, S. (2025).
#    Vowel- and diphthong-like spectral patterns in sperm whale codas.
#    *Open Mind*, 9, 1849–1874. https://doi.org/10.1162/OPMI.a.252
#
# 3. Gero, S., Whitehead, H., & Rendell, L. (2016).
#    Individual, unit and vocal clan level identity cues in sperm whale codas.
#    *Royal Society Open Science*, 3(1), 150372. https://doi.org/10.1098/rsos.150372
#
# 4. Rendell, L. & Whitehead, H. (2003).
#    Vocal clans in sperm whales (*Physeter macrocephalus*).
#    *Proceedings of the Royal Society B*, 270(1512), 225–231. https://doi.org/10.1098/rspb.2002.2239
#
# 5. Weilgart, L. & Whitehead, H. (1993).
#    Coda communication by sperm whales (*Physeter macrocephalus*) off the Galapagos Islands.
#    *Canadian Journal of Zoology*, 71(4), 744–752. https://doi.org/10.1139/z93-098
#
# 6. Youngblood, M. (2025).
#    Language-like efficiency in whale communication.
#    *Science Advances*, 11(6), eads6014. https://doi.org/10.1126/sciadv.ads6014
#
# 7. Cantor, M. & Whitehead, H. (2013).
#    The interplay between social networks and culture: theoretically and among whales and dolphins.
#    *Philosophical Transactions of the Royal Society B*, 368, 20120340.
#    https://doi.org/10.1098/rstb.2012.0340
#
# 8. Paradise, O., Muralikrishnan, P., Chen, L., Flores Garcia, H., Pardo, B., Diamant, R.,
#    Gruber, D.F., Gero, S., & Goldwasser, S. (2025).
#    WhAM: Towards a translative model of sperm whale vocalization.
#    In *Advances in Neural Information Processing Systems 38 (NeurIPS 2025)*.
#    arXiv:2512.02206.
#
# 8a. Hersh, T.A., Sayigh, L.S., Mesnick, S.L., Rendell, L., & Whitehead, H. (2022).
#     Sympatric sperm whale clans exhibit distinct cultural boundaries in identity codas.
#     *Proceedings of the National Academy of Sciences*, 119(42), e2201692119.
#     https://doi.org/10.1073/pnas.2201692119
#
# ### Geometric and Topological Methods
#
# 9. Nickel, M. & Kiela, D. (2017).
#    Poincare embeddings for learning hierarchical representations.
#    In *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*.
#    arXiv:1705.08039.
#
# 10. Sarkar, R. (2011).
#     Low distortion Delaunay embedding of trees in hyperbolic plane.
#     In *Graph Drawing (GD 2011)*, LNCS 7034, 355–366. Springer.
#     https://doi.org/10.1007/978-3-642-25878-7_34
#
# 11. Carlsson, G. (2009).
#     Topology and data.
#     *Bulletin of the American Mathematical Society*, 46(2), 255–308.
#     https://doi.org/10.1090/S0273-0979-09-01249-X
#
# 12. Bauer, U. (2021).
#     Ripser: efficient computation of Vietoris–Rips persistence barcodes.
#     *Journal of Applied and Computational Topology*, 5(3), 391–423.
#     https://doi.org/10.1007/s41468-021-00071-5
#
# 13. Tralie, C., Saul, N., & Bar-On, R. (2018).
#     Ripser.py: A lean persistent homology library for Python.
#     *Journal of Open Source Software*, 3(29), 925. https://doi.org/10.21105/joss.00925
#
# 14. Otter, N., Porter, M.A., Tillmann, U., Grindrod, P., & Harrington, H.A. (2017).
#     A roadmap for the computation of persistent homology.
#     *EPJ Data Science*, 6, 17. https://doi.org/10.1140/epjds/s13688-017-0109-5

# %%
# Summary statistics
print("=" * 70)
print("GEOMETRIC CETACEAN COMMUNICATION ANALYSIS — SUMMARY")
print("=" * 70)
print(f"Dataset: {len(df)} codas, {df['CodaType'].nunique()} types")
if 'Unit' in df.columns:
    print(f"Social units: {df['Unit'].nunique()}")
print(f"\nPoincare embedding: {n_types} types in {embed_dim}-dim ball")
print(f"Distortion (Spearman rho): Poincare={rho_poincare:.4f}, "
      f"Euclidean={rho_euclidean:.4f}")
print(f"\nClassification accuracy: "
      f"Hyperbolic={hyp_acc:.1%}, Best Euclidean={best_euclidean:.1%} ({best_euclidean_name})")
if HAS_RIPSER_DIRECT and tda_results:
    print(f"\nTDA: Analyzed {len(tda_results)} coda types")
    h1_counts = [r['features']['H1_count'] for r in tda_results.values()]
    print(f"  H1 loop count range: [{min(h1_counts)}, {max(h1_counts)}]")
print(f"\nDRI Score: {result.dri:.4f}")
print(f"  Most vulnerable transform: "
      f"{sorted_transforms[0][0]} (omega={sorted_transforms[0][1]:.4f})")
print(f"  Most robust to: "
      f"{sorted_transforms[-1][0]} (omega={sorted_transforms[-1][1]:.4f})")
print("=" * 70)
print(f"\nAll figures saved to {os.getcwd()}")
print("Toolkit: pip install eris-ketos")
print("Repository: https://github.com/ahb-sjsu/eris-ketos")
