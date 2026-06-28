
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from boruta import BorutaPy   # maintained version


# methods
def calc_bmi(row):
    if pd.notnull(row['Gewicht']) and pd.notnull(row['Grösse']):
        bmi = row['Gewicht'] / (row['Grösse'] / 100)**2
        return bmi
    return np.nan


def calc_lambda(row):
    """Estimate the effective decay rate (lambda) based on the daily dose rate measurements
        using linear regression on the linearized data."""
    time_h = [0]
    log_activities = []
    dose_rate_labels = ['D0 calc', 'D1', 'D2', 'D3', 'D4', 'D5']
    time_labels = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5']
    # change time format from hh.mm to hh.decimalhour
    for i, label in enumerate(time_labels):
        t = row[label]
        if t > 0:
            hours = int(t)
            mins = (t - hours) / 60 * 100    # convert from mm to hh
            if i == 0:
                t0 = hours + mins
            else:
                time = (hours + mins + i * 24) - t0    # interval of measurements: every day
                time_h.append(time)
            log_activities.append(np.log(row[dose_rate_labels[i]]))
            # do not break loop if a value is missing (might be a gap, not the end ;)
    if len(time_h) >= 2:
        # fit log data where -lambda_effective is the slope: D' = D'_0 * exp(-lambda_eff * t)
        slope, intercept, r_val, _, _ = linregress(time_h, log_activities)
        return -slope, r_val**2
    else:
        return np.nan, np.nan


def predict_discharge_time(row, predicted_lambda):
    """Predict duration [h] until residual dose rate falls below threshold = 10 uSv/h/m."""
    current_dose = row['D0 calc']
    if current_dose <= 10:
        return 0    # already discharged
    hours_needed = (np.log(current_dose) - np.log(10)) / predicted_lambda
    return float(hours_needed)


def discharge_classification(row):
    """Create row with binary data for the 48h discharge criteria."""
    return 1 if row['discharge_hour'] < 48 else 0



### LOAD DATA ###
sheet_name = 'M'
name_dict = {'B': 'Benign', 'M': 'Malignant'}
best_threshold = None    # for later (track back false-positives)

df = pd.read_excel('Radioiodine_data.xlsx', sheet_name=sheet_name)
print(f'Sheet data is extracted from:   {sheet_name}')
cols = ', '.join(df.columns)
print(f'{"Column names:":<20}{cols}')

# clean column names and categorical data and choose predictors (features available before therapy results)
df.rename(columns={'A0 [MBq]': 'A0', 'D1 [uSv/h]': 'D1', 'Vol [ml]': 'Vol'}, inplace=True)    # rename columns with problematic characters for XGBoost
cols = ', '.join(df.columns)
print(f'{"New column names:":<20}{cols:>100}')


### PREPARE PREDICTORS ###
# df_clean = df_clean.dropna(subset=['eGFR lab'])     # only take true eGRF in this run
df['gender_binary'] = df['G'].map({'M': 0, 'W': 1})     # translate categorical to binary data by returning a scalar to every element of a df
df['BMI'] = df.apply(calc_bmi, axis=1, result_type='expand')

# define features (predictors) to be used for the analysis
if sheet_name == 'M':   # malign
    df['rhTSH_binary'] = df['rhTSH'].map({'n': 0, 'y': 1})
    df['M-local'] = (((df['N'].str.contains('0')) | (df['N'].str.contains('X'))) & (df['M'].str.contains('0'))).astype(int)
    df['M-meta'] = df['meta_flag'] = (~df['N'].astype(str).str.contains('0|X').fillna(False)).astype(int)
    features = ['A0', 'Alter', 'rhTSH_binary', 'BMI', 'gender_binary', 'eGFR', 'Gewicht', 'Grösse', 'M-meta', 'M-local', 'KSB']

if sheet_name == 'B':   # benign
    if 'Patho' in df.columns:
        patho_dummies = pd.get_dummies(df['Patho'], prefix='Patho').astype(int)     # one-hot encoding (separate rows for each pathology with 0 or 1)
        df = pd.concat([df, patho_dummies], axis=1)
    features = ['A0', 'Alter', 'Gewicht', 'eGFR', 'gender_binary', 'Vol', 'Up 24', 'Up 48', 'Grösse', 'KSB', 'Patho_MB', 'Patho_FA', 'Patho_MA', 'Patho_BA', 'Patho_DA']

if sheet_name == 'total':
    patho_dummies = pd.get_dummies(df['Patho'], prefix='Patho').astype(int)     # one-hot encoding (separate rows for each pathology with 0 or 1)
    df = pd.concat([df, patho_dummies], axis=1)
    df['Patho_B'] = (patho_dummies['Patho_FA'] | patho_dummies['Patho_BA'] | patho_dummies['Patho_MA'] | patho_dummies['Patho_DA'] | patho_dummies['Patho_MB']).astype(int)
    features = ['A0', 'Alter', 'Gewicht', 'eGFR', 'gender_binary', 'Patho_B', 'KSB']


### DISCHARGE DURATION ###

seed = 13

# estimate effective decay constant (lambda_eff) and exclude datasets if required
df[['lambda_eff', 'fit_R2']] = df.apply(calc_lambda, axis=1, result_type='expand')
df_clean = df.dropna(subset=features).copy()
df_clean = df.dropna(subset=['lambda_eff'])
X = df_clean[features]
print(f'Dataset ({sheet_name}) reduced from {len(df)} to {len(X)} patients after verifying complete data sets.')
X = X.dropna()
rows_to_drop = X.index[X.isna().any(axis=1)]    # drop rows with missing data

# create classification variable: duration > / < 48h
indices = X.index
d0_values = df_clean.loc[indices, 'D0 calc']
t0_values = df_clean.loc[indices, 'T0']
calc_df = pd.DataFrame({
    'D0 calc': d0_values,
    'T0': t0_values})
df_clean['discharge_hour'] = df_clean.apply(lambda row: predict_discharge_time(row, predicted_lambda=row['lambda_eff']), axis=1)
df_clean['discharge_48'] = df_clean.apply(discharge_classification, axis=1, result_type='expand')

y = df_clean['discharge_48']
y = y.loc[X.index]

print(f'Input features: {X.columns}')


### NESTED CV ###

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

outer_scores = []
outer_preds = []
outer_probs = []
outer_true = []
fold_selected_features = []  # track selected features per outer fold
false_positive_idx = []

for train_idx, test_idx in outer_cv.split(X, y):

    # outer-fold train/validation split
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # instantiate new preprocessing, selector, model for each outer fold
    preprocess_cv = ColumnTransformer(transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories='auto'), categorical_cols),
        ],
        remainder='drop'
    )

    rf_for_boruta_cv = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        class_weight='balanced_subsample',
        n_jobs=1,            # Boruta compatibility
        random_state=seed
    )

    boruta_cv = BorutaPy(
        estimator=rf_for_boruta_cv,
        n_estimators='auto',
        perc=100,
        max_iter=100,
        two_step=True,
        random_state=seed,
        verbose=0
    )

    model_cv = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced_subsample',
        n_jobs=-1,
        random_state=seed
    )

    pipe_cv = Pipeline(steps=[
            ('prep', preprocess_cv),
            ('boruta', boruta_cv),
            ('clf', model_cv),
        ]
    )

    # inner CV for feature selection (scores required for hyperparameter tuning)
    # _ = cross_val_score(pipe_cv, X_train_fold, y_train_fold, cv=inner_cv, scoring='roc_auc', n_jobs=1)

    # fit pipeline on the full outer-training fold (incl. Boruta) and record selected features
    pipe_cv.fit(X_train_fold, y_train_fold)
    feat_names_fold = pipe_cv.named_steps['prep'].get_feature_names_out()
    mask_fold = pipe_cv.named_steps['boruta'].support_
    selected_fold = feat_names_fold[mask_fold]
    print(selected_fold)
    fold_selected_features.append(selected_fold.tolist())

    # predict on the outer test fold (unseen by selection)
    y_pred_fold = pipe_cv.predict(X_test_fold)
    y_prob_fold = pipe_cv.predict_proba(X_test_fold)[:, 1]

    outer_preds.extend(y_pred_fold)
    outer_probs.extend(y_prob_fold)
    outer_true.extend(y_test_fold)
    fold_auc = roc_auc_score(y_test_fold, y_prob_fold)
    outer_scores.append(fold_auc)

    if best_threshold:
        fp_mask = (y_test_fold == 0) & (y_prob_fold > best_threshold)
        fp_original_idx = test_idx[fp_mask]    # map back to original dataset indices
        false_positive_idx.extend(fp_original_idx)


print(f'Nested CV AUC: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}')

# feature selection stability summary
all_selected = [f for fold in fold_selected_features for f in fold]
freq = Counter(all_selected)
print('Feature selection frequency across outer folds (count):')
for feat, count in freq.most_common():
    print(f'  {feat}: {count}')


### FIT FINAL PIPELINE ON X ###

# instantiate fresh final pipeline (do not reuse fitted objects)
preprocess_final = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories='auto'), categorical_cols),
        ], remainder='drop'
)

rf_for_boruta_final = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight='balanced_subsample',
    n_jobs=1,
    random_state=seed
)

boruta_final = BorutaPy(
    estimator=rf_for_boruta_final,
    n_estimators='auto',
    perc=100,
    max_iter=100,
    two_step=True,
    random_state=seed,
    verbose=0
)

model_final = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=seed
)

pipe_final = Pipeline(steps=[
        ('prep', preprocess_final),
        ('boruta', boruta_final),
        ('clf', model_final),
    ]
)

pipe_final.fit(X, y)
