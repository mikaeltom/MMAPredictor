from .AbstractAgent import AbstractAgent
import os
import pandas as pd
from datetime import datetime
import numpy as np


class HistoricalAgent(AbstractAgent):
    """
    Agent that analyses the historical data of both fighters. The idea
    is to first observe the head to head between the two fighters. By analyzing
    their confrontation against each other, we can observe if one has a
    psychological advantage over the other one as the psychological state of fighter
    impact its performance. This head-to-head analysis is made by giving a
    head-to-head score that we designed. The intuition behind this score is to give weights
    to their confrontation. For instance, a win in 2016 has not the same impact as
    a win against the same fighter one year ago. Thus, we use the negative exponential of
    the year that passed since the fights. Using this Exponential Decay gives more power
    to the recent fights. After that, current form of the fighter is important. We retrieve the
    fighter last win streak (how much win without loosing) and the amount of days since
    the fighter did not lose.
    """
    def __init__(self, name):
        """
        Initializes the historical data that are used by the agent
        """
        super().__init__(name)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_name = os.path.join(root, "Data", "historical_data.csv") # source of data : https://www.kaggle.com/datasets/rajaisrarkiani/ufc-fights-and-fighter-stats-dataset?select=Fights+Data.csv

    def process_data(self):
        """
        Loads, clean the historical data and convert the dat in right format.
        The date equals the number of days elapsed since each fight.
        """
        df = pd.read_csv(self.db_name)
        df["event_date"] = df["event_date"].str.replace(".", "") # source : https://www.geeksforgeeks.org/data-analysis/python-pandas-dataframe-replace/
        df["event_date"] = pd.to_datetime(df["event_date"], format="mixed") # source : https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
        df = df[["fight_fighter", "opponent", "fight_result", "event_date"]] #source : https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
        today = datetime.today()  # source : https://stackoverflow.com/questions/32490629/getting-todays-date-in-yyyy-mm-dd-in-python
        df["event_date"] = (today - df["event_date"]).dt.days  # source : https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.days.html
        return df

    def retrieve_head_to_head_data(self, df, fighter1, fighter2):
        """
        Filters the name of the given fighters, to retrieve their past fights
        against each other.Since the data file contains duplicate lines, for instance
        a line (fighter1, fighter2, win, date) with a line fighter2, fighter1, loose, date),
        we only keep the winning lines. So we avoid counting two times the fights.
        """
        head_to_heads = df[((df["fight_fighter"] == fighter1) & (df["opponent"] == fighter2)) |  # source : https://www.kdnuggets.com/2022/12/five-ways-conditional-filtering-pandas.html
                ((df["fight_fighter"] == fighter2) & (df["opponent"] == fighter1))]
        head_to_heads = head_to_heads[head_to_heads["fight_result"] == "win"] # since all fights are there two times in the db (fighter1, fighter2) and (fighter2, fighter2), juste keep the winner
        head_to_heads =  head_to_heads.sort_values(by="event_date", ascending=True) # source : https://www.geeksforgeeks.org/pandas/how-to-sort-pandas-dataframe/
        if head_to_heads.empty:
            return None # if no historical fight between them
        return head_to_heads

    def compute_head_to_head_score(self,  df, fighter1, fighter2):
        """
        Method that gives the amount of wins for each fighter in their preivious
        fights against each other. It also computes the head to head score for each fighter.
        As explained, this score is an exponential decay of the number of days elapsed
        since each fight.
        """
        data = self.retrieve_head_to_head_data(df, fighter1, fighter2)
        if data is None:
            return None, None, None, None, None
        f1_win = 0 # number of win in their confrontation
        f2_win = 0
        f1_score = 0.0
        f2_score = 0.0
        for _, fight in data.iterrows(): # source : https://www.geeksforgeeks.org/pandas/different-ways-to-iterate-over-rows-in-pandas-dataframe/
            day_since_fight = fight["event_date"]
            if np.isnan(day_since_fight): # source : # source : https://www.geeksforgeeks.org/python/python-pandas-dataframe-isna/
                w = np.exp(-1.0) # default value in case NaN
            else:
                w = np.exp(-day_since_fight / 365)  # source : https://en.wikipedia.org/wiki/Exponential_decay
            if fight["fight_fighter"] == fighter1: # since the data is ordered in way that first fighter = name of the winner
                f1_win += 1
                f1_score += w
            else:
                f2_win += 1
                f2_score += w
        return "valid", f1_win, f2_win, f1_score, f2_score

    def response_head_to_head_score(self, fighter1, fighter2, fighter1_h2h, fighter2_h2h, fighter1_score, fighter2_score):
        """
        Generates readable summary of the head to head score for each fighter in
        their previous fights against each other. This will be given as prompt
        to the LLM that will make the final choice.
        """
        if fighter1_score >= fighter2_score:
            winner, winner_h2h, winner_score = fighter1, fighter1_h2h, fighter1_score
            looser, looser_h2h, looser_score = fighter2, fighter2_h2h, fighter2_score
        else:
            winner, winner_h2h, winner_score = fighter2, fighter2_h2h, fighter2_score
            looser, looser_h2h, looser_score = fighter1, fighter1_h2h, fighter1_score
        return (f"Comparing Head to Head between {fighter1} and {fighter2}, we can see that Fighter {winner} won "
                f"{winner_h2h} of the {winner_h2h + looser_h2h} fights, while {looser} won only {looser_h2h} of them."
                f"This gives a Head to Head score of {winner_score} for {winner} and a score of {looser_score} for {looser}.")

    def retrieve_win_streak(self, df, fighter):
        """
        Calculates the last consecutive win streak of the given fighter.
        A win streak is when the fighter wins all its game. So here, we increment
        from its last fight, until the fighter lost a game.
        """
        fights = df[(df["fight_fighter"] == fighter)] # only one column since database duplicated each row, so just need to check 1 col
        fights = fights.dropna(subset=["event_date"]) # source : https://www.geeksforgeeks.org/python/python-pandas-dataframe-dropna/
        win_streak = fights.sort_values(by="event_date",ascending=True)
        if win_streak.empty:
            return None, None
        win = 0
        streak_date = None # date of the start of the win streak
        for _, fight in win_streak.iterrows():
            if fight["fight_result"] == "win":
                win += 1
                streak_date = fight["event_date"]
            else:
                break
        return win, streak_date

    def response_fighter_streak(self, df, fighter):
        """
        Generates human readable summary of the win streak of the given fighter.
        """
        win, streak_date = self.retrieve_win_streak(df, fighter)
        if win is None:
            return f"No information available about the winning streak of {fighter}\n"
        if win == 0 :
            return f"{fighter} has no winning streak right now.\n"
        return f"{fighter} won its last {win} fights. {fighter} is in a {int(streak_date)} day winning streak.\n"

    def process(self, *args):
        """
        Processes the head-to-head scoring computation and the win streak
        analysis, to produce a final summary containing the two analysis.
        """
        fighter1 = args[0]
        fighter2 = args[1]
        df = self.process_data()
        status_h2h, fighter1_h2h, fighter2_h2h, fighter1_score, fighter2_score = self.compute_head_to_head_score(df, fighter1, fighter2)
        response = ""
        if status_h2h is None:
            response += f"No Head to Head Historical Data for the fights between {fighter1} and {fighter2}\n"
        else:
            response += self.response_head_to_head_score(fighter1, fighter2, fighter1_h2h, fighter2_h2h, fighter1_score, fighter2_score)
        response += self.response_fighter_streak(df, fighter1)
        response += self.response_fighter_streak(df, fighter2)
        return response
