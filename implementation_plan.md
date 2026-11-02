# Myopia Prediction – Implementation Plan

## Dataset Overview
- **X_train**: 132 samples × 49 features (mix of numerical and categorical)
- **Y_train**: 132 samples × 4 label columns (M, MM, Combo, DCombo)
- **X_test**: 24 samples × 49 features (no labels — must predict)

## Key Observations
| Column | Classes | Distribution |
|--------|---------|--------------|
| **M** (Myopia) | SI/NO | 88 / 44 (2:1 imbalance) |
| **MM** (Myopia Magna) | SI/NO | 10 / 122 (heavy imbalance) |
| **Combo** | M / C / MM | 70 / 44 / 18 |
| **DCombo** | C / M1 / M2 / MM | 44 / 39 / 31 / 18 |

> [!IMPORTANT]
> If a person has **Myopia Magna (MM=SI)**, they automatically have **Myopia (M=SI)**. This logical constraint must be enforced in post-processing.

## Phase Plan

### Phase 1 – Data Understanding & EDA
- Load data, inspect shapes, dtypes, missing values
- Visualize target distributions (bar charts)
- Correlation heatmap for numerical features
- Box plots / violin plots for key features vs targets

### Phase 2 – Data Cleaning & Feature Engineering
- **Drop**: `fecha` (date, not predictive), `origen antepasados (españa)` and `origen antepasados (extranjeros)` (too many unique values + many nulls), `hᵃ act.cerca sem` (100% missing)
- **Encode** categorical columns: `sexo`, `Exterior Mucho`, `Interior Mcho`, `Familiar MM SI/NO`, `fumador`, `toma sol`, `protege sol`, `disp.elec.antes dormir`, `fototipo`, `Grupo fot`, `familiar miope`, `familiar miope magno`, `pat. Ret. Miop magna`
- **Impute** missing numerical values (median) and categorical (mode)
- **Feature importance** analysis with Random Forest to identify most useful features

### Phase 3 – Model Training & Evaluation
For each of the 4 targets, train:
1. **Random Forest** (baseline, robust to mixed features)
2. **Gradient Boosting (HistGradientBoosting)** (handles missing values natively, often better accuracy)
3. **SVM** (good for small datasets)

Evaluation:
- Stratified K-Fold cross-validation (k=5)
- Metrics: Accuracy, F1-score (macro), confusion matrix
- Select best model per target

### Phase 4 – Final Predictions & Post-processing
- Train best model on full training set
- Predict on X_test
- **Enforce constraint**: If MM=SI → force M=SI
- Save predictions to TXT file

### Phase 5 – Summary & Conclusions
- Summary table of all model results
- Feature importance plots
- Final remarks
