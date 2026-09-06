import pandas as pd

FEATURES_TO_KEEP = [ # features on the 95 we have to keep (we tested to avoid leakage)
    "reach_diff",
    "age_diff",
    "height_diff",
    "SLpM_total_diff",
    "SApM_total_diff",
    "sig_str_acc_total_diff",
    "str_def_total_diff",
    "td_avg_diff",
    "td_acc_total_diff",
    "td_def_total_diff",
    "sub_avg_diff",
    "wins_total_diff",
    "losses_total_diff",
]

def process_raw_fight_data():
    """
    Function that preprocesses the dataset contained the data to train the xgboost model.
    This dataset comes from : https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024
    There are 95 features, after analysing and multiple attempts to avoid data leakage, we kept the features in
    FEATURES_TO_KEEP as well as the winner.
    """
    path_to_data = "large_dataset.csv"
    df = pd.read_csv(path_to_data, low_memory=False)
    df = df[df["winner"].isin(["Red", "Blue"])] # filter possible draws
    df["target"] = (df["winner"] == "Red").astype(int) # red win = 1, blue win = 0 as historically red corner = favorite
    df = df[FEATURES_TO_KEEP + ["target"]]
    df = df.dropna(subset=["reach_diff"]) # this is an important feature, so if it is not there we remove the total line
    df = df.fillna(df.median(numeric_only=True)) # source : https://medium.com/@whee.2013/pandas-data-cleaning-guide-how-to-clean-transform-and-prepare-data-for-analysis-3c921f5a3e2f
    df.to_csv("preprocessed_data_for_training.csv", index=False)

process_raw_fight_data()
