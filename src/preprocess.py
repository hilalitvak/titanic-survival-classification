import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

KEEP_TITLES = {'Mr', 'Miss', 'Mrs', 'Master'}

FEATURES = [
    'Pclass', 'Age', 'FamilySize', 'fare_per_person', 'has_cabin',
    'age_child', 'age_adult', 'age_senior',
    'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare',
    'Embarked_C', 'Embarked_Q', 'Embarked_S',
]


def _extract_titles(df: pd.DataFrame) -> pd.Series:
    """Extract titles from Name and normalise to 5 groups.
    Titles with fewer than 10 occurrences are grouped into Rare — too few samples to learn from reliably."""
    title = df['Name'].str.extract(r',\s*([^.]+)\.')[0].str.strip()
    title = title.replace({'Mlle': 'Miss', 'Mme': 'Mrs', 'Ms': 'Miss'})
    return title.apply(lambda t: t if t in KEEP_TITLES else 'Rare')


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out['Pclass']          = df['Pclass']
    out['Age']             = df['Age']
    out['FamilySize']      = df['SibSp'] + df['Parch'] + 1
    out['has_cabin']       = df['Cabin'].notna().astype(int)
    out['fare_per_person'] = df['Fare'] / out['FamilySize']
    out['Title']           = _extract_titles(df)
    out['Embarked']        = df['Embarked']
    return out


def _add_age_groups(out: pd.DataFrame) -> pd.DataFrame:
    """Add age group dummies after Age has been imputed."""
    out['age_child']  = (out['Age'] < 15).astype(int)
    out['age_senior'] = (out['Age'] > 60).astype(int)
    out['age_adult']  = ((out['Age'] >= 15) & (out['Age'] <= 60)).astype(int)
    return out


def fit_transform(df: pd.DataFrame):
    """Fit preprocessing on df (training set) and return transformed X, y, and params dict."""
    out = _build_features(df)

    # Impute — fit values on training data only
    age_by_title  = out.groupby('Title')['Age'].median()
    embarked_mode = out['Embarked'].mode()[0]

    out['Age']      = out.apply(
        lambda r: age_by_title[r['Title']] if pd.isna(r['Age']) else r['Age'], axis=1
    )
    out['Embarked'] = out['Embarked'].fillna(embarked_mode)

    out = _add_age_groups(out)
    out = pd.get_dummies(out, columns=['Title', 'Embarked'], drop_first=False)

    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0

    X = out[FEATURES].values.astype(np.float32)
    y = df['Survived'].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    params = {
        'age_by_title':  age_by_title.to_dict(),
        'embarked_mode': embarked_mode,
        'scaler':        scaler,
    }
    return X, y, params


def transform(df: pd.DataFrame, params: dict) -> np.ndarray:
    """Apply pre-fitted params to a new dataframe (validation / inference)."""
    out = _build_features(df)

    age_by_title  = params['age_by_title']
    embarked_mode = params['embarked_mode']

    out['Age'] = out.apply(
        lambda r: age_by_title.get(r['Title'], np.median(list(age_by_title.values())))
        if pd.isna(r['Age']) else r['Age'],
        axis=1,
    )
    out['Embarked'] = out['Embarked'].fillna(embarked_mode)

    out = _add_age_groups(out)
    out = pd.get_dummies(out, columns=['Title', 'Embarked'], drop_first=False)

    for col in FEATURES:
        if col not in out.columns:
            out[col] = 0

    X = out[FEATURES].values.astype(np.float32)
    return params['scaler'].transform(X)
