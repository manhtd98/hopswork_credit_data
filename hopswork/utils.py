import pandas as pd


def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    nan_df = df.isna().sum()[df.isna().sum() > 0]
    nan_perc = (nan_df / df.shape[0] * 100).apply(int)
    cols_to_drop = nan_perc[nan_perc >= 20].index
    df = df.drop(cols_to_drop, axis=1).dropna()
    return df


def one_hot_encoder(df, nan_as_category=False):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(
        df, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True
    )
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
