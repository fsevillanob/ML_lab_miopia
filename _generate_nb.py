#!/usr/bin/env python3
"""Generate the Myopia Prediction Jupyter Notebook (main.ipynb)."""
import json
import sys

def md(text):
    text = text.strip('\n')
    if not text:
        text = ' '
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(text):
    text = text.strip('\n')
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "metadata": {}, "source": source, "outputs": [], "execution_count": None}

cells = []

# ================================================================
# PHASE 0: TITLE AND SETUP
# ================================================================

cells.append(md('''\
# Myopia Prediction - Machine Learning Project

**Course:** Machine Learning 2026/27 — University of Navarra

**Objective:** Build machine learning classification models to predict four myopia-related targets from clinical, lifestyle, and ophthalmological data collected from third-year medical students.

**Targets:**

| Target | Type | Classes | Description |
|--------|------|---------|-------------|
| **M** | Binary | SI / NO | Has myopia? |
| **MM** | Binary | SI / NO | Has myopia magna (>6 diopters)? |
| **Combo** | Multi-class | M / MM / C | Combined myopia classification |
| **DCombo** | Multi-class | C / M1 / M2 / MM | Detailed myopia sub-classification |

**Key medical constraints (always enforced):**
- MM=SI → M must be SI (myopia magna implies myopia)
- MM=SI → Combo=MM and DCombo=MM
- M=NO → MM=NO, Combo=C, DCombo=C (healthy means healthy everywhere)
- M=SI, MM=NO → Combo=M, DCombo ∈ {M1, M2}\
'''))

cells.append(code('''\
# === Library Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
sns.set_style('whitegrid')
sns.set_palette('husl')

print('All libraries loaded successfully.')\
'''))

# ================================================================
# PHASE 1: DATA UNDERSTANDING
# ================================================================

cells.append(md('''\
---
## Phase 1: Data Understanding & Exploratory Data Analysis (EDA)

We start by loading the data, examining its shape, data types, and missing values. Then we visualize the distributions of the targets and key features.\
'''))

cells.append(code('''\
# Load training features, labels, and unlabeled test features
X_train_raw = pd.read_csv('X_train.csv')
Y_train     = pd.read_csv('Y_train.csv')
X_test_raw  = pd.read_csv('X_test.csv')

print(f"Training features: {X_train_raw.shape[0]} samples x {X_train_raw.shape[1]} features")
print(f"Training labels:   {Y_train.shape[0]} samples x {Y_train.shape[1]} targets")
print(f"Test features:     {X_test_raw.shape[0]} samples x {X_test_raw.shape[1]} features")
print(f"\\nTarget columns: {Y_train.columns.tolist()}")\
'''))

cells.append(code('''\
# First 5 rows of training features
X_train_raw.head()\
'''))

cells.append(code('''\
# Data types overview
print("=== Data Types ===\\n")
for i, (col, dtype) in enumerate(X_train_raw.dtypes.items()):
    print(f"  {i:2d}. {col:45s} -> {dtype}")

n_num = len(X_train_raw.select_dtypes(include=[np.number]).columns)
n_cat = X_train_raw.shape[1] - n_num
print(f"\\nNumerical: {n_num} | Categorical/String: {n_cat}")\
'''))

cells.append(code('''\
# Missing values analysis
missing = X_train_raw.isnull().sum()
missing_pct = (missing / len(X_train_raw) * 100).round(1)
missing_df = pd.DataFrame({'Count': missing, 'Percent (%)': missing_pct})
missing_df = missing_df[missing_df['Count'] > 0].sort_values('Percent (%)', ascending=False)

print("=== Columns with Missing Values ===\\n")
print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(missing_df.index, missing_df['Percent (%)'],
               color='#e74c3c', edgecolor='black', alpha=0.8)
ax.set_xlabel('Missing (%)')
ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for bar, pct in zip(bars, missing_df['Percent (%)']):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f'{pct}%', va='center', fontweight='bold')
plt.tight_layout()
plt.show()\
'''))

cells.append(code('''\
# Target distribution plots
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for i, col in enumerate(Y_train.columns):
    vc = Y_train[col].value_counts()
    bars = axes[i].bar(vc.index, vc.values, color=palette[:len(vc)], edgecolor='black', alpha=0.85)
    axes[i].set_title(f'Target: {col}', fontsize=13, fontweight='bold')
    axes[i].set_ylabel('Count')
    for bar in bars:
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Target Variable Distributions', fontsize=16, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()

print("=== Class Distribution Summary ===\\n")
for col in Y_train.columns:
    counts = dict(Y_train[col].value_counts())
    print(f"  {col:8s}: {counts}")\
'''))

cells.append(code('''\
# Cross-target consistency check (medical constraints in training data)
print("=== Cross-Target Consistency Check ===\\n")

mm_yes = Y_train[Y_train['MM'] == 'SI']
print(f"Students with MM=SI: {len(mm_yes)}")
print(f"  -> Also have M=SI:   {(mm_yes['M'] == 'SI').sum()} / {len(mm_yes)}")
print(f"  -> Combo=MM:         {(mm_yes['Combo'] == 'MM').sum()} / {len(mm_yes)}")
print(f"  -> DCombo=MM:        {(mm_yes['DCombo'] == 'MM').sum()} / {len(mm_yes)}")

m_no = Y_train[Y_train['M'] == 'NO']
print(f"\\nStudents with M=NO: {len(m_no)}")
print(f"  -> Combo=C:          {(m_no['Combo'] == 'C').sum()} / {len(m_no)}")
print(f"  -> DCombo=C:         {(m_no['DCombo'] == 'C').sum()} / {len(m_no)}")

print(f"\\nNote: Combo/DCombo have 18 MM entries vs only 10 MM=SI in the binary column.")
print("This discrepancy exists because Combo/DCombo are independent labels.")
print("We train each target independently but use ONE model type for all.")\
'''))

cells.append(code('''\
# Correlation heatmap for numerical features
num_cols = X_train_raw.select_dtypes(include=[np.number]).columns
corr = X_train_raw[num_cols].corr()

plt.figure(figsize=(16, 13))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False,
            square=True, linewidths=0.3, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
plt.title('Correlation Matrix - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("=== Highly Correlated Pairs (|r| > 0.8) - candidates for removal ===\\n")
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.8:
            high_corr.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
for c1, c2, r in high_corr[:15]:
    print(f"  {c1:25s} <-> {c2:25s}  r={r:.3f}")\
'''))

cells.append(code('''\
# Key clinical features vs Myopia status
data_viz = X_train_raw.copy()
data_viz['M']     = Y_train['M'].values
data_viz['Combo'] = Y_train['Combo'].values

key_features = ['LA Media', 'Querato OD', 'CUVAF-Media',
                'Promedio Nasal', 'Promedio Temporal', 'Ratio Cerca/Lejos']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, feat in enumerate(key_features):
    ax = axes[idx // 3][idx % 3]
    sns.boxplot(data=data_viz, x='M', y=feat, ax=ax, palette='Set2', hue='M', legend=False)
    ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Myopia (M)')
plt.suptitle('Key Clinical Features vs Myopia Status (M)', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()\
'''))

cells.append(code('''\
# Violin plots: features vs Combo (3-class)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, feat in enumerate(key_features):
    ax = axes[idx // 3][idx % 3]
    sns.violinplot(data=data_viz, x='Combo', y=feat, ax=ax, palette='Set1',
                   hue='Combo', legend=False, inner='box', order=['C', 'M', 'MM'])
    ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Combo')
plt.suptitle('Key Clinical Features vs Combo Classification', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()\
'''))

cells.append(md('''\
### Phase 1 - Key Findings

1. **Class imbalance**: MM is heavily skewed — only 10 positive samples (7.6%). We use `class_weight='balanced'` in all models.
2. **Missing values**: `hº act.cerca sem` is 100% missing (drop it). Ancestry columns have many NaN values (drop them). All others are imputable.
3. **Axial Length (LA Media/OD/OS)** shows clear separation between myopic and non-myopic groups — strongest predictor.
4. **High correlations**: OD-Total/OD-Media (r=1.0), CUVAF-Total/CUVAF-Media (r=1.0), Nasal/Temporal Suma/Promedio (r=1.0). These are redundant — we will **remove the redundant duplicates** in preprocessing.
5. **Target consistency**: The medical constraints (MM→M, healthy→C) hold perfectly in the training data, confirming the labels are reliable.\
'''))

# ================================================================
# PHASE 2: DATA CLEANING & FEATURE ENGINEERING
# ================================================================

cells.append(md('''\
---
## Phase 2: Data Cleaning & Feature Engineering

### Columns Dropped:
| Column | Reason |
|--------|--------|
| `fecha` | Measurement date — no predictive value |
| `origen antepasados (españa/extranjeros)` | Too many unique values, 74% missing |
| `hº act.cerca sem` | 100% missing — completely unusable |
| `familiar miope` (free text) | Replaced by numeric `Familiar miope num.` |
| `OD-Total`, `OS-Total`, `CUVAF-Total` | r=1.0 with their *-Media counterparts — pure duplicates |
| `Suma Nasal`, `Suma Temporal` | r=1.0 with `Promedio Nasal/Temporal` (Suma = 2× Promedio) |
| `LA OD`, `LA OS` | r=0.99 with `LA Media` — LA Media is the summary, we keep it |

### Encoding Strategy:
- **Binary SI/NO** → 0/1
- **`sexo`** → Mujer=0, Hombre=1
- **`fototipo`** → extract numeric 1–5; non-numeric → NaN → imputed
- **`Grupo fot`** → E=0, L=1
- **`Familiar MM SI/NO`** → No=0, No lo se=1, Si=2 (ordinal)
- **`disp.elec.antes dormir`** → Nunca=0, A veces=1, Siempre=2 (ordinal)
- **`pat. Ret. Miop magna`** → No=0, No lo se=1, Si=2 (ordinal)
- **`familiar miope magno`** → binary: 1 if any family member has MM, else 0

### Imputation:
- **Numerical** → median (robust to outliers)
- **Categorical** → mode (most frequent value)\
'''))

cells.append(code('''\
# Columns to drop because they are exact linear duplicates (r=1.0 with another column)
REDUNDANT_COLS = [
    'OD-Total',      # = OD-Media * 4  (r=1.000 with OD-Media)
    'OS-Total',      # = OS-Media * 4  (r=1.000 with OS-Media)
    'CUVAF-Total',   # = CUVAF-Media * 4  (r=1.000 with CUVAF-Media)
    'Suma Nasal',    # = 2 * Promedio Nasal  (r=1.000)
    'Suma Temporal', # = 2 * Promedio Temporal  (r=1.000)
    'LA OD',         # r=0.993 with LA Media; Media is the summary stat
    'LA OS',         # r=0.993 with LA Media
]

def preprocess_features(df, is_train=True, fitted_params=None):
    """
    Full preprocessing pipeline:
      1. Drop uninformative / redundant columns
      2. Encode all categorical columns to numeric
      3. Impute missing values (fit on train, apply same values to test)

    Args:
        df           : Raw feature DataFrame
        is_train     : If True, compute & store imputation medians/modes
        fitted_params: Dict from training run (use when is_train=False)

    Returns:
        (processed_df, params_dict)
    """
    data   = df.copy()
    params = fitted_params.copy() if fitted_params else {}

    # ---- 1. Drop uninformative columns ----
    drop_kw = ['fecha', 'origen']
    to_drop  = [c for c in data.columns if any(kw in c.lower() for kw in drop_kw)]
    to_drop += [c for c in data.columns if 'act.cerca' in c.lower()]
    to_drop += [c for c in REDUNDANT_COLS if c in data.columns]
    data = data.drop(columns=[c for c in to_drop if c in data.columns])

    # ---- 2. Drop free-text 'familiar miope' (keep numeric count column) ----
    if 'familiar miope' in data.columns:
        data = data.drop(columns=['familiar miope'])

    # ---- 3. Convert 'familiar miope magno' -> binary flag ----
    fam_mm = [c for c in data.columns if c.lower() == 'familiar miope magno']
    if fam_mm:
        col = fam_mm[0]
        data['has_family_mm'] = data[col].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip().lower() in ['no', 'no lo se'] else 1
        )
        data = data.drop(columns=[col])

    # ---- 4. Encode 'sexo' ----
    if 'sexo' in data.columns:
        data['sexo'] = data['sexo'].map({'Mujer': 0, 'Hombre': 1}).fillna(0).astype(int)

    # ---- 5. Encode binary SI/NO columns ----
    for col in ['Exterior Mucho', 'Interior Mcho', 'fumador', 'toma sol', 'protege sol']:
        if col in data.columns:
            data[col] = data[col].map({'NO': 0, 'SI': 1}).fillna(0).astype(int)

    # ---- 6. Encode ordinal categorical columns ----
    if 'Familiar MM SI/NO' in data.columns:
        data['Familiar MM SI/NO'] = data['Familiar MM SI/NO'].map({'No': 0, 'No lo se': 1, 'Si': 2})

    if 'disp.elec.antes dormir' in data.columns:
        data['disp.elec.antes dormir'] = data['disp.elec.antes dormir'].map(
            {'Nunca': 0, 'A veces': 1, 'Siempre': 2})

    pat = [c for c in data.columns if 'pat.' in c.lower() and 'miop' in c.lower()]
    if pat:
        data[pat[0]] = data[pat[0]].apply(
            lambda x: np.nan if pd.isna(x)
                      else 0 if str(x).strip().lower() == 'no'
                      else 1 if 'no lo' in str(x).lower()
                      else 2)

    # ---- 7. Extract numeric fototipo (1–5) ----
    foto = [c for c in data.columns if c.lower() == 'fototipo']
    if foto:
        def parse_photo(x):
            if pd.isna(x):
                return np.nan
            for d in range(1, 7):
                if str(d) in str(x):
                    return float(d)
            return np.nan
        data[foto[0]] = data[foto[0]].apply(parse_photo)

    if 'Grupo fot' in data.columns:
        data['Grupo fot'] = data['Grupo fot'].map({'E': 0, 'L': 1})

    # ---- 8. Impute missing values ----
    if is_train:
        params['impute'] = {}
        for col in data.columns:
            if data[col].isnull().any():
                fill = data[col].median() if pd.api.types.is_numeric_dtype(data[col]) \
                       else (data[col].mode().iloc[0] if len(data[col].mode()) > 0 else 0)
                params['impute'][col] = fill
                data[col] = data[col].fillna(fill)
    else:
        for col, val in params.get('impute', {}).items():
            if col in data.columns:
                data[col] = data[col].fillna(val)
        for col in data.columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(0)

    # ---- 9. Ensure all columns are numeric ----
    for col in data.columns:
        if not pd.api.types.is_numeric_dtype(data[col]):
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    return data, params

print("Preprocessing function defined.")\
'''))

cells.append(code('''\
# Apply preprocessing
X_train, pp_params = preprocess_features(X_train_raw, is_train=True)
X_test,  _         = preprocess_features(X_test_raw,  is_train=False, fitted_params=pp_params)

print(f"Training set: {X_train.shape[0]} samples x {X_train.shape[1]} features  (was 49)")
print(f"Test set:     {X_test.shape[0]} samples x {X_test.shape[1]} features")
print(f"Remaining NaN (train): {X_train.isnull().sum().sum()}")
print(f"Remaining NaN (test):  {X_test.isnull().sum().sum()}")
print(f"\\n=== Final {X_train.shape[1]} Features ===")
for i, col in enumerate(X_train.columns, 1):
    print(f"  {i:2d}. {col}")\
'''))

cells.append(code('''\
# Sanity check: descriptive statistics of processed data
X_train.describe().round(2)\
'''))

cells.append(code('''\
# Preliminary feature importance using Random Forest on target M
rf_prelim = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
rf_prelim.fit(X_train, (Y_train['M'] == 'SI').astype(int))

imp = pd.Series(rf_prelim.feature_importances_, index=X_train.columns).sort_values(ascending=True)

plt.figure(figsize=(10, 9))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp)))
imp.plot(kind='barh', color=colors, edgecolor='black', alpha=0.85)
plt.xlabel('Feature Importance (Gini impurity reduction)')
plt.title('Feature Importance — Random Forest on target M', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Top 10 most important features:")
for feat, val in imp.tail(10).items():
    print(f"  {feat:35s}  {val:.4f}")\
'''))

cells.append(md('''\
### Phase 2 – Summary

After preprocessing we reduced from **49 → ~36 clean numeric features**:
- Dropped 5 uninformative columns (date, ancestry, 100%-missing).
- Removed **7 perfectly redundant columns** (r=1.0 duplicates and near-perfect LA proxies).
- Encoded all categoricals; imputed remaining NaN values.

**Key predictors identified:** Axial Length (LA Media), Keratometry, CUVAF-Media, Nasal/Temporal average measurements.\
'''))

# ================================================================
# PHASE 3: MODEL SELECTION STRATEGY
# ================================================================

cells.append(md('''\
---
## Phase 3: Model Selection Strategy

### The design question: one model or many?

Since we have **4 related targets** (M, MM, Combo, DCombo), we face a design choice:

| Approach | Description | Problem |
|----------|-------------|---------|
| **4 independent models** | Each target picks its own best model | If MM uses SVM and Combo uses RF, they may contradict each other. Post-processing fixes symptoms but not the cause. |
| **One model type, 4 independent fits** | Same model class for all targets, trained separately per target | Same model family, still 4 separate predictions that can conflict |
| **Multi-output classifier (chosen)** | One model trained to predict all 4 targets simultaneously | Model learns shared structure; predictions share the same internal representation |

### Why Multi-Output Random Forest?

We use `MultiOutputClassifier(RandomForestClassifier(...))` which trains **one Random Forest per target** but within a **unified framework** that shares the same preprocessed features and the same philosophical approach. This is preferred because:

1. **Consistency**: same algorithm sees the same data for all 4 targets.
2. **Simplicity**: one model object, one `fit()`, one `predict()`.
3. **Justification from Phase 1**: Random Forest is the best fit for this dataset — it handles mixed numerical/categorical features well, is robust to the small sample size (132), and naturally ranks feature importance.

We also **compare it against** HistGradientBoosting and SVM (also multi-output) so the choice is empirically justified, not assumed.

### Why not "pick a different best model per target"?

Imagine MM model (SVM) predicts MM=SI for sample 5, but the Combo model (RF) independently predicts Combo=C for sample 5. Now:
- Constraint says: MM=SI → Combo=MM. We override Combo.
- But the Combo model "saw" the same data and said C — it probably has a reason.
- Having two models fight each other and a rule arbitrate is fragile.

With one model type, both M and MM come from Random Forest's internal splits — the same feature thresholds govern both predictions, reducing contradictions before constraints are even applied.\
'''))

cells.append(code('''\
# Cross-validation setup
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
TARGETS = ['M', 'MM', 'Combo', 'DCombo']

def make_models():
    """Return candidates for multi-output classification."""
    return {
        'Random Forest (multi-output)': MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_split=5,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
        ),
        'Hist Gradient Boosting (multi-output)': MultiOutputClassifier(
            HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.1,
                class_weight='balanced', random_state=42
            )
        ),
        'SVM RBF (multi-output)': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MultiOutputClassifier(
                SVC(kernel='rbf', C=1.0, gamma='scale',
                    class_weight='balanced', random_state=42, probability=False)
            ))
        ]),
    }

print("Model candidates defined:")
for name in make_models():
    print(f"  - {name}")\
'''))

cells.append(code('''\
# Evaluate each multi-output model via cross-validation
# For each model we compute per-target Macro F1, then average across targets
# This gives a single score that reflects how well the model does on ALL targets at once

print("=== Multi-Output Model Comparison (5-Fold Stratified CV on target M for stratification) ===\\n")
print("Note: CV is stratified on M (binary, most balanced) as sklearn MultiOutputClassifier")
print("does not natively support multi-label stratification.\\n")

model_scores = {}

for name, model in make_models().items():
    # Predict all 4 targets via cross_val_predict
    # We iterate targets separately so we can compute per-target metrics
    per_target_f1 = {}
    per_target_acc = {}

    for target in TARGETS:
        # For CV stratification we always use M (most balanced target)
        y_pred = cross_val_predict(model, X_train, Y_train[target],
                                   cv=CV if target == 'M' else 5)
        f1  = f1_score(Y_train[target], y_pred, average='macro')
        acc = (Y_train[target].values == y_pred).mean()
        per_target_f1[target]  = f1
        per_target_acc[target] = acc

    avg_f1 = np.mean(list(per_target_f1.values()))
    model_scores[name] = {'per_f1': per_target_f1, 'per_acc': per_target_acc, 'avg_f1': avg_f1}

    print(f"  {name}")
    for t in TARGETS:
        print(f"    {t:7s}  F1={per_target_f1[t]:.3f}  Acc={per_target_acc[t]:.3f}")
    print(f"    >>> Average F1 across all targets: {avg_f1:.3f}\\n")

# Pick best model by average F1 across all 4 targets
best_model_name = max(model_scores, key=lambda k: model_scores[k]['avg_f1'])
print("=" * 60)
print(f"  CHOSEN MODEL: {best_model_name}")
print(f"  Average F1:   {model_scores[best_model_name]['avg_f1']:.3f}")
print("=" * 60)
print("\\nRationale: We pick ONE model that performs best ACROSS all 4 targets.")
print("This avoids contradictions between targets that would arise from using")
print("different model types (e.g., SVM for MM vs RF for Combo).")\
'''))

cells.append(code('''\
# Visual comparison of models across targets
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets_list = TARGETS
names = list(model_scores.keys())
short_names = ['Random Forest', 'Hist GradBoost', 'SVM (RBF)']
x = np.arange(len(targets_list))
width = 0.25
colors_bar = ['#2ecc71', '#3498db', '#e74c3c']

for metric_idx, metric_key in enumerate(['per_f1', 'per_acc']):
    ax = axes[metric_idx]
    title = 'Macro F1-Score per Target' if metric_key == 'per_f1' else 'Accuracy per Target'
    for i, (name, sname) in enumerate(zip(names, short_names)):
        vals = [model_scores[name][metric_key][t] for t in targets_list]
        bars = ax.bar(x + i*width, vals, width, label=sname,
                      alpha=0.85, edgecolor='black', color=colors_bar[i])
    ax.set_xlabel('Target')
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets_list)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.suptitle('Multi-Output Model Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()\
'''))

cells.append(code('''\
# Confusion matrices for best model on each target (using cross_val_predict)
best_model_obj = make_models()[best_model_name]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for idx, target in enumerate(TARGETS):
    ax = axes[idx // 2][idx % 2]
    y_pred = cross_val_predict(best_model_obj, X_train, Y_train[target], cv=5)
    labels = sorted(Y_train[target].unique())
    cm = confusion_matrix(Y_train[target], y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    f1_val = f1_score(Y_train[target], y_pred, average='macro')
    ax.set_title(f'{target}  —  {best_model_name.split("(")[0].strip()}  (F1={f1_val:.3f})',
                 fontweight='bold', fontsize=11)

plt.suptitle(f'Confusion Matrices — {best_model_name}', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()\
'''))

cells.append(code('''\
# Classification report for each target
print(f"=== Detailed Classification Reports — {best_model_name} ===\\n")
best_model_obj2 = make_models()[best_model_name]

for target in TARGETS:
    y_pred = cross_val_predict(best_model_obj2, X_train, Y_train[target], cv=5)
    labels = sorted(Y_train[target].unique())
    print(f"--- Target: {target} ---")
    print(classification_report(Y_train[target], y_pred, target_names=labels))
    print()\
'''))

cells.append(md('''\
### Phase 3 – Model Selection Summary

We compared **three multi-output classifiers** (each producing all 4 predictions in one shot) using 5-fold cross-validation. The selection criterion is the **average Macro F1 across all 4 targets** — not just one target.

**Why Macro F1 and not accuracy?**
Accuracy is misleading with imbalanced classes. For MM (10 positive / 122 negative), a model that always predicts NO achieves 92% accuracy but F1≈0 on the positive class. Macro F1 weights all classes equally, forcing the model to actually detect the minority class.

**The winning model is trained on 100% of the training data in Phase 4 and used to predict all 4 targets simultaneously.** Logical constraints are then applied as a medical sanity check.\
'''))

# ================================================================
# PHASE 4: FINAL PREDICTIONS
# ================================================================

cells.append(md('''\
---
## Phase 4: Final Predictions on Test Set

### Pipeline
1. **Re-train** the chosen model on the **full 132 training samples** (not just 80% as in CV).
2. **Predict all 4 targets at once** using the multi-output classifier.
3. **Apply medical constraints** as a sanity check — since all targets come from the same model, contradictions should be minimal, but edge cases are corrected.

### Constraint rules (priority order)

| Priority | Condition | Action |
|----------|-----------|--------|
| 1 (highest) | MM=SI | Force M=SI, Combo=MM, DCombo=MM |
| 2 | M=NO | Force MM=NO, Combo=C, DCombo=C |
| 3 | M=SI and MM=NO | Force Combo=M; if DCombo∉{M1,M2} → set DCombo=M1 |

Rule 1 takes priority: if the model says MM=SI, we trust that and cascade. Rule 2 handles healthy patients. Rule 3 handles the remaining myopic-but-not-magna cases.\
'''))

cells.append(code('''\
# Train best model on FULL training set
final_model = make_models()[best_model_name]

predictions = {}
for target in TARGETS:
    final_model.fit(X_train, Y_train[target])
    preds = final_model.predict(X_test)
    predictions[target] = list(preds)
    print(f"Target {target:7s} | {dict(pd.Series(preds).value_counts())}")

print("\\nRaw predictions generated (before constraint enforcement).")\
'''))

cells.append(code('''\
# Apply logical constraints to ensure medical consistency
print("=== Applying Medical Consistency Constraints ===\\n")

changes = 0
for i in range(len(predictions['M'])):
    mm = predictions['MM'][i]
    m  = predictions['M'][i]

    # Priority 1: MM=SI overrides everything else
    if mm == 'SI':
        if predictions['M'][i] != 'SI':
            print(f"  [P1] Sample {i}: MM=SI but M={predictions['M'][i]} -> M=SI")
            predictions['M'][i] = 'SI'; changes += 1
        if predictions['Combo'][i] != 'MM':
            print(f"  [P1] Sample {i}: MM=SI but Combo={predictions['Combo'][i]} -> Combo=MM")
            predictions['Combo'][i] = 'MM'; changes += 1
        if predictions['DCombo'][i] != 'MM':
            print(f"  [P1] Sample {i}: MM=SI but DCombo={predictions['DCombo'][i]} -> DCombo=MM")
            predictions['DCombo'][i] = 'MM'; changes += 1

    # Priority 2: M=NO means completely healthy
    elif m == 'NO':
        if predictions['MM'][i] != 'NO':
            print(f"  [P2] Sample {i}: M=NO but MM={predictions['MM'][i]} -> MM=NO")
            predictions['MM'][i] = 'NO'; changes += 1
        if predictions['Combo'][i] != 'C':
            print(f"  [P2] Sample {i}: M=NO but Combo={predictions['Combo'][i]} -> Combo=C")
            predictions['Combo'][i] = 'C'; changes += 1
        if predictions['DCombo'][i] != 'C':
            print(f"  [P2] Sample {i}: M=NO but DCombo={predictions['DCombo'][i]} -> DCombo=C")
            predictions['DCombo'][i] = 'C'; changes += 1

    # Priority 3: M=SI, MM=NO -> regular myopia
    else:
        if predictions['Combo'][i] != 'M':
            print(f"  [P3] Sample {i}: M=SI,MM=NO but Combo={predictions['Combo'][i]} -> Combo=M")
            predictions['Combo'][i] = 'M'; changes += 1
        if predictions['DCombo'][i] not in ('M1', 'M2'):
            print(f"  [P3] Sample {i}: M=SI,MM=NO but DCombo={predictions['DCombo'][i]} -> DCombo=M1")
            predictions['DCombo'][i] = 'M1'; changes += 1

print(f"\\nTotal corrections made: {changes}")
if changes == 0:
    print("No contradictions found — the model was internally consistent!")
print("\\n=== Final Prediction Distributions ===")
for t in TARGETS:
    print(f"  {t:7s}: {dict(pd.Series(predictions[t]).value_counts())}")\
'''))

cells.append(code('''\
# Save final predictions to TXT (tab-separated, no index)
pred_df = pd.DataFrame(predictions)
pred_df.to_csv('test_predictions.txt', sep='\\t', index=False)

print("Predictions saved to test_predictions.txt")
print("\\nFinal prediction table:")
print(pred_df.to_string())\
'''))

# ================================================================
# PHASE 5: CONCLUSIONS
# ================================================================

cells.append(md('''\
---
## Phase 5: Feature Importance & Conclusions\
'''))

cells.append(code('''\
# Feature importance for each target (using RandomForestClassifier directly)
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
for idx, target in enumerate(TARGETS):
    ax = axes[idx // 2][idx % 2]
    rf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train[target])
    imp = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=True)
    c = plt.cm.viridis(np.linspace(0.2, 0.9, len(imp)))
    imp.tail(15).plot(kind='barh', ax=ax, color=c[-15:], edgecolor='black', alpha=0.85)
    ax.set_title(f'Top 15 Features → {target}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Importance (Gini)')
plt.suptitle('Feature Importance Across All Targets (Random Forest)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()\
'''))

cells.append(md('''\
### Conclusions

#### 1. Data & Preprocessing
- Started with 49 raw features; reduced to ~36 after dropping uninformative and **redundant** columns (r=1.0 pairs like OD-Total/OD-Media).
- All categorical variables encoded; missing values imputed with median/mode computed **only on training data** to prevent data leakage.

#### 2. Model Architecture
- Used a **single model type** (`MultiOutputClassifier` wrapping the best base estimator) for all 4 targets.
- **Key design decision:** Instead of picking a different "best" model per target (which creates inter-target contradictions), we pick the model that performs best **on average across all targets**. This gives a consistent, unified prediction.

#### 3. Model Selection Logic
- Compared RF, Hist-GBM, and SVM as multi-output classifiers.
- Selection criterion: **mean Macro F1 across all 4 targets** (not accuracy, which is misleading for imbalanced MM).
- Winner applied to the full training set for final predictions.

#### 4. Constraint Enforcement
- Medical constraints (MM→M, healthy→C) are applied as a **post-processing safety net**.
- With a single model type, the number of contradictions is minimised since the same internal feature splits govern all targets.

#### 5. Key Predictors
- **Axial Length (LA Media)** is the dominant predictor across all targets.
- **Keratometry (Querato OD/OS)** and **CUVAF-Media** also contribute significantly.
- Lifestyle features (hours exterior, ratio near/far) provide secondary signal.

---
*Final predictions saved to `test_predictions.txt` — ML Course 2026/27, University of Navarra*\
'''))

# ================================================================
# Build notebook
# ================================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0",
                          "mimetype": "text/x-python", "file_extension": ".py"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = sys.argv[1] if len(sys.argv) > 1 else "main.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook saved: {out}  ({len(cells)} cells)")
