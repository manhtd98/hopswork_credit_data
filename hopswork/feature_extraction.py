import pandas as pd
import numpy as np
import hopsworks
from utils import *
import config

# ignore warnings
import warnings
import slugify

warnings.filterwarnings("ignore")


def age_cat(df):
    interval = (18, 25, 35, 60, 120)

    cats = ["Student", "Young", "Adult", "Senior"]
    df["Age_cat"] = pd.cut(df.Age, interval, labels=cats)
    return df


def preprocessing_data(df_credit):
    df_credit["Saving accounts"] = df_credit["Saving accounts"].fillna("no_inf")
    df_credit["Checking account"] = df_credit["Checking account"].fillna("no_inf")
    df_credit = age_cat(df_credit)
    # Purpose to Dummies Variable
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit.Purpose, drop_first=True, prefix="Purpose"),
        left_index=True,
        right_index=True,
    )
    # Sex feature in dummies
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit.Sex, drop_first=True, prefix="Sex"),
        left_index=True,
        right_index=True,
    )
    # Housing get dummies
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit.Housing, drop_first=True, prefix="Housing"),
        left_index=True,
        right_index=True,
    )
    # Housing get Saving Accounts
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit["Saving accounts"], drop_first=True, prefix="Savings"),
        left_index=True,
        right_index=True,
    )
    # Housing get Risk
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit.Risk, prefix="Risk"), left_index=True, right_index=True
    )
    # Housing get Checking Account
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit["Checking account"], drop_first=True, prefix="Check"),
        left_index=True,
        right_index=True,
    )
    # Housing get Age categorical
    df_credit = df_credit.merge(
        pd.get_dummies(df_credit["Age_cat"], drop_first=True, prefix="Age_cat"),
        left_index=True,
        right_index=True,
    )
    # Excluding the missing columns
    del df_credit["Saving accounts"]
    del df_credit["Checking account"]
    del df_credit["Purpose"]
    del df_credit["Sex"]
    del df_credit["Housing"]
    del df_credit["Age_cat"]
    del df_credit["Risk"]
    del df_credit["Risk_good"]

    df_credit["Credit amount"] = np.log(df_credit["Credit amount"])

    df_credit.columns = [
        slugify.slugify(x).replace("-", "_") for x in df_credit.columns
    ]
    df_credit["credit_id"] = df_credit.index.values
    return df_credit


if __name__ == "__main__":
    project = hopsworks.login()
    fs = project.get_feature_store()
    applications_df = pd.read_csv(config.CONFIG.DF_PATH, index_col=0)
    applications_df = preprocessing_data(applications_df)
    applications_fg = fs.get_or_create_feature_group(
        name=config.CONFIG.FEATURE_GROUP,
        version=1,
        online_enabled=False,
        primary_key=["credit_id"],
    )

    applications_fg.insert(
        applications_df,
        write_options={"wait_for_job": True},
    )
    print("Success to extract feautures and upload!!!")
