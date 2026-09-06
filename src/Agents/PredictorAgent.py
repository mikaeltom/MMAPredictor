from .AbstractAgent import AbstractAgent
import json
import os
import pickle
import pandas as pd
import dice_ml
import numpy as np
from datetime import datetime


HARMONIZED_NAMES = { # both stats dataset have different names, we keep only 1
    "splm": "SLpM", "str_acc" : "sig_str_acc", "sapm" : "SApM", "td_avg_acc": "td_acc",
}

CONTINUOUS_FEATURE = [ # keep only the features that are continuous number
        "reach_diff", "age_diff", "height_diff",
        "SLpM_total_diff", "SApM_total_diff", "sig_str_acc_total_diff",
        "str_def_total_diff", "td_avg_diff", "td_acc_total_diff",
        "td_def_total_diff", "sub_avg_diff", "wins_total_diff", "losses_total_diff",
]

DIFF_TO_STATS = { # convert the difference stats used in xgboost to "classical" columns
        "reach_diff": "reach",
        "age_diff": "age",
        "height_diff": "height",
        "SLpM_total_diff": "SLpM",
        "SApM_total_diff": "SApM",
        "sig_str_acc_total_diff": "sig_str_acc",
        "str_def_total_diff": "str_def",
        "td_avg_diff": "td_avg",
        "td_acc_total_diff": "td_acc",
        "td_def_total_diff": "td_def",
        "sub_avg_diff": "sub_avg",
        "wins_total_diff": "wins",
        "losses_total_diff":"losses",
}

STATS_TO_NAME = { # for explanation, covert table columns to real descriptions
    "reach" : "Reach",
    "age" : "Age",
    "height" :"Height",
    "SLpM" : "Significant Strikes Landed per Minute",
    "SApM": "Significant Strikes Absorbed per Minute",
    "sig_str_acc" : "Significant Striking Accuracy",
    "str_def" : "Striking Defense Efficiency",
    "td_avg" : "Average Takedowns Landed",
    "td_acc" : "Takedown Accuracy",
    "td_def" : "Takedown Defense Efficiency",
    "sub_avg" : "Average Submission Attempts",
    "wins" : "Wins",
    "losses" : "Losses",
}

FEATURES_TO_VARY = [ # features that can change to give action to looser (so not age, height,...)
    "reach_diff",
    "SLpM_total_diff", "SApM_total_diff", "sig_str_acc_total_diff",
    "str_def_total_diff", "td_avg_diff", "td_acc_total_diff",
    "td_def_total_diff", "sub_avg_diff", "wins_total_diff", "losses_total_diff",
]

class PredictorAgent(AbstractAgent):
    """
    PredictorAgent is an agent that uses the XGBOOST trained model
    to predict the winner of the asker fight, while explaining the
    results using counterfactuals. We chose XGBOOST as it is widely
    considered as the "standard" for predictions in sports. The idea
    of the explainability in our system is to make local explanations
    using Diverse Counterfactual Explanation (DiCE), as seen in classes.
    The intuition is to explain to the loosers, what feature should they
    improve in order to become the predicted winners. For that, we propose
    3 scenario of actions and the final LLM will decide which one to keep.
    We use the library : https://github.com/interpretml/dice for DiCE.
    """
    def __init__(self, name):
        """
        Initialize the datasets used by the agent and the model and DiCE
        for explainability.
        """
        super().__init__(name)
        self.model, self.model_columns, self.df_stats, self.df_stats_backup, self.df_training = self.setup_agent()
        self.dice = self.setup_explanablility()

    def setup_agent(self):
        """
        Loads the Models and the datasets used by the agent.
        """
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "Models", "xgb_model.pkl")
        features_order_path = os.path.join(project_root, "Models", "features.json")
        with open(model_path, "rb") as f:
            model = pickle.load(f) # source : https://www.geeksforgeeks.org/machine-learning/saving-and-loading-xgboost-models/
        with open(features_order_path, "r") as f: # order of features to be sure we respect the same format as xgboost train
            model_columns = json.load(f) # source https://docs.python.org/3/library/json.html

        fighter_stats_path = os.path.join(project_root, "Data", "fighter_recent.csv") # source : https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025?select=fighter_details.csv
        fighter_stats_backup_path = os.path.join(project_root, "Data", "fighter_stats.csv") # source : https://www.kaggle.com/datasets/maksbasher/ufc-complete-dataset-all-events-1996-2024
        training_data = os.path.join(project_root, "Data", "preprocessed_data_for_training.csv")
        df_stats = pd.read_csv(fighter_stats_path)
        df_stats_backup = pd.read_csv(fighter_stats_backup_path)
        df_training = pd.read_csv(training_data)
        df_stats["name"] = df_stats["name"].str.strip().str.lower() # normalize the name in lower case
        df_stats_backup["name"] = df_stats_backup["name"].str.strip().str.lower()
        return model, model_columns, df_stats, df_stats_backup, df_training

    def setup_explanablility(self):
        """
        Method that initializes the Diverse Counterfactual Explanation (DICE)
        This code is inspired by : https://github.com/interpretml/dice
        """
        columns = CONTINUOUS_FEATURE + ["target"]
        d = dice_ml.Data(dataframe=self.df_training[columns], continuous_features=CONTINUOUS_FEATURE, outcome_name="target")
        m = dice_ml.Model(model=self.model, backend="sklearn") # source : https://interpret.ml/DiCE/notebooks/DiCE_getting_started.html
        exp = dice_ml.Dice(d, m, method="genetic")
        return exp

    def convert_to_age(self, df):
        """
        Convert a date in an age. Example : Aug 01, 1985 becomes 40.
        """
        df["dob"] = pd.to_datetime(df["dob"], format='%b %d, %Y') # source : https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
        today = datetime.today() # source : https://stackoverflow.com/questions/32490629/getting-todays-date-in-yyyy-mm-dd-in-python
        df["dob"] = (today - df["dob"]).dt.days // 365 # source : https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days.html
        df = df.rename(columns={"dob": "age"})
        return df

    def find_fighter_stats(self, name):
        """
        Find the fighters stats in the databases. There are two databases,
        we check in the most recent, if the stats are not there we
        use fixed database, since scrapping website did not work.
        We were blocked from website. We could use API to retrieve stats,
        but there is no free API, they all requires a memberships. The
        most recent database is 3 month old, and since fighters do not
        fight every moth (since they have a bg time of recovery) we assumed
        that these stats are enough for our system. In addition, we checked
        before arriving here that the fighters exists and that the fight
        between them exists, so no need to check that here.
        """
        name = name.strip()
        name = name.lower()
        stats = self.df_stats[self.df_stats["name"] == name].copy() # copy avoid pandas warning
        if not stats.empty:
            stats = self.convert_to_age(stats)
            return stats.rename(columns=HARMONIZED_NAMES) # source : https://stackoverflow.com/questions/11346283/renaming-column-names-in-pandas
        else: # backup if not found in first
            stats = self.df_stats_backup[self.df_stats_backup["name"] == name].copy()
            return stats

    def convert_to_float(self, v):
        """
        Safely convert a string to float to avoid errors.
        """
        if pd.isna(v): # source : https://www.geeksforgeeks.org/python/python-pandas-dataframe-isna/
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError): # source : https://www.geeksforgeeks.org/python/float-in-python/
            return 0.0

    def compute_difference(self, fighter1_stats, fighter2_stats, col_name):
        """
        Calculate the raw difference between the fighter for a given feature.
        Since the first fighter is the one in the "red corner", so the favorite,
        as in the data used for training xgboost, the second fighter is
        subtracted from the first fighter.
        """
        val1 = fighter1_stats[col_name].iloc[0] # source : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iloc.html
        val2 = fighter2_stats[col_name].iloc[0]
        val1 = self.convert_to_float(val1)
        val2 = self.convert_to_float(val2)
        return val1 - val2

    def get_dictionnary_of_differences(self, fighter1_stats, fighter2_stats):
        """
        Computes the difference between fighter1 and fighter2 in all the features
        that are used for training xgboost. It returns that as a dictionary.
        """
        return {
            "reach_diff": self.compute_difference(fighter1_stats, fighter2_stats, "reach"),
            "age_diff": self.compute_difference(fighter1_stats, fighter2_stats, "age"),
            "height_diff": self.compute_difference(fighter1_stats, fighter2_stats, "height"),
            "SLpM_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "SLpM"),
            "SApM_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "SApM"),
            "sig_str_acc_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "sig_str_acc"),
            "str_def_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "str_def"),
            "td_avg_diff": self.compute_difference(fighter1_stats, fighter2_stats, "td_avg"),
            "td_acc_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "td_acc"),
            "td_def_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "td_def"),
            "sub_avg_diff": self.compute_difference(fighter1_stats, fighter2_stats, "sub_avg"),
            "wins_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "wins"),
            "losses_total_diff": self.compute_difference(fighter1_stats, fighter2_stats, "losses"),
        }

    def get_features_difference_vector(self, fighter1_stats, fighter2_stats):
        """
        Given the dictionary of differences between fighter1 and fighter2,
        it creates a DataFrame that will be used by the model to predict.
        """
        differences = self.get_dictionnary_of_differences(fighter1_stats, fighter2_stats)
        df = pd.DataFrame([differences]) # source : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
        df = df[self.model_columns] # source : https://www.geeksforgeeks.org/python/change-the-order-of-a-pandas-dataframe-columns-in-python/
        return df

    def explain_counterfactual(self, counter_examples, original_data, winner_stats, loser_stats):
        """
        Method that uses DiCE explanation in terms of difference and convert them to the user
        real features, for each scenario of actions. Since model operates on a difference
        vector for predicting, DiCE gives the counterfactuals in terms of differences.
        We take this differences and convert them to  fighter features value. We achieve that
        by doing : Modified_Looser_Value = Winner Value - DiCE action value (in terms of difference).
        Since there are multiple actions by scenarios, we are keeping the 3 most important differences
        to changes as other small changes may be noise. It is a personal choice to avoid the final
        decision maker LLM having too much actions and hallucinate.
        """
        scenarios = []
        for i in range(len(counter_examples)):
            counterfact = counter_examples.iloc[i]
            changes = []
            differences = []
            for feature in DIFF_TO_STATS:
                difference = counterfact[feature] - original_data[feature]
                if abs(difference) > 0.01:  # threshold to assume that there is a change
                    stats_column = DIFF_TO_STATS[feature]
                    val_winner = self.convert_to_float(winner_stats[stats_column].iloc[0])
                    val_loser = self.convert_to_float(loser_stats[stats_column].iloc[0])
                    new_val_loser = val_winner - float(counterfact[feature]) # get the user value by doing winner feature - counterfactual difference feature
                    sign = "higher" if new_val_loser > val_loser else "lower"
                    if new_val_loser < 0:
                        continue
                    changes.append(f"{STATS_TO_NAME[stats_column]} {sign} ({val_loser:.2f} to {new_val_loser:.2f})")
                    differences.append(abs(difference))
            if changes: # if counterfactuals
                max_indices = np.argsort(differences)[-3:]  # source : https://www.geeksforgeeks.org/python/sort-sorteda-np-argsorta-np-lexsortb-python/
                max_indices = list(max_indices)
                max_indices.reverse() # we take the 3 most important differences, since it generates a lot of actions, we keep 3 bests
                ouptut_changes = [changes[i] for i in max_indices]
                to_output = ', '.join(ouptut_changes)  # source : https://www.geeksforgeeks.org/python/python-string-join-method/
                scenarios.append(f"Scenario {i + 1} : {to_output}")
        if len(scenarios) > 0:
            return scenarios
        else:
            return ["No Counterfactual"]

    def generate_counterfactual(self, prediction_data, prediction_output, winner_stats, loser_stats):
        """
        Uses DiCE library to generate 3 different scenarios of diverse counterfactuals
        explanations. It formats this explanation and return tha as a list of scenarios.
        """
        desired_output = 1 - prediction_output # desired output is the opposite
        prediction_data_continuous = prediction_data[CONTINUOUS_FEATURE]
        top3_counter = self.dice.generate_counterfactuals(prediction_data_continuous, total_CFs=3,
                        desired_class=desired_output, features_to_vary=FEATURES_TO_VARY) # source : https://interpret.ml/DiCE/notebooks/DiCE_model_agnostic_CFs.html
        counter_examples = top3_counter.cf_examples_list[0].final_cfs_df # source : https://github.com/interpretml/DiCE/issues/174
        original_data = prediction_data.iloc[0]
        return self.explain_counterfactual(counter_examples, original_data, winner_stats, loser_stats)

    def process(self, *args):
        """
        Principal Method. It runs the prediction pipeline , by fetching
        the statistics, computing the difference vector, training that model
        on the difference vector, predicting the outcome and use the retrieve
        the confidence of the model, finally, generates the counterfactual
        explanations to guide the LLM that will do the final choice.
        """
        fighter1_name = args[0]
        fighter2_name = args[1]
        fighter1_stats = self.find_fighter_stats(fighter1_name)
        fighter2_stats = self.find_fighter_stats(fighter2_name)

        prediction_data = self.get_features_difference_vector(fighter1_stats, fighter2_stats) # vector in format : np.array([[1.2, 3.4, 5.6]])
        prediction_confidence = self.model.predict_proba(prediction_data) # source : https://stackoverflow.com/questions/70945378/select-one-probability-out-of-two-from-xgboost-returned-predict-proba
        prediction = int(np.argmax(prediction_confidence[0])) # prediction is a list of two probabilities, the highest is the winner
        confidence = prediction_confidence[0][prediction] # probability that a certain fighter is the winner
        winner, winner_stats = (fighter1_name, fighter1_stats) if prediction == 1 else (fighter2_name, fighter2_stats)
        loser, loser_stats = (fighter2_name, fighter2_stats) if prediction == 1 else (fighter1_name, fighter1_stats)

        counter_factual = self.generate_counterfactual(prediction_data, prediction, winner_stats, loser_stats)

        return f"{winner} wins with {confidence*100:.1f}% confidence. Counterfactual : {loser} could have won if {counter_factual}"
