import sqlite3
from datetime import datetime, timedelta
import os

class PredictionCache:
    """
    Manages the prediction cache represented as an sqlite3
    database. The objective of this cache is to minimize
    redundant LLM and API calls as the requested are limited.
    So, predictions are cached in a database four 24 hours,
    after this delay, the fight is not considered as cached
    anymore, fresh prediction can be made again on this fight.
    """
    def __init__(self):
        """
        Initializes the database file path
        """
        root = os.path.dirname(os.path.abspath(__file__))
        self.db_name = os.path.join(root, "Data", "predictioncache.db")
        self.init_db()

    def init_db(self):
        """
        Creates the table that will hold the prediction cache,
        if not already existing. It contains the two fighter names,
        the prediction in text and the date of prediction.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fighter1 TEXT,
            fighter2 TEXT,
            prediction TEXT,
            last_updated TEXT NOT NULL,
            UNIQUE(fighter1, fighter2)
            )
            """)
            conn.commit()

    def retrieve_predictions(self, fighter_1, fighter_2):
        """
        Attempts to retrieve the predictions for the fight between
        fighter1 and fighter2 and checks if it was made less than
        24 hours ago. If it is the case it returns it, otherwise, this
        prediction is out-of-date (more than 24 hours) and cannot be
        retrieved.
        """
        with sqlite3.connect(self.db_name) as conn:
            for fighter1, fighter2 in [(fighter_1, fighter_2), (fighter_2, fighter_1)]: # check both possibilities
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT prediction, last_updated 
                    FROM predictions 
                    WHERE fighter1 = ? AND fighter2 = ?
                """, (fighter1, fighter2))
                row = cursor.fetchone()

                if row:
                    prediction, last_updated_str = row
                    last_updated = datetime.fromisoformat(last_updated_str) # source : https://www.geeksforgeeks.org/python/fromisoformat-function-of-datetime-date-class-in-python/
                    if datetime.now() - last_updated <= timedelta(hours=24): # source : https://www.influxdata.com/blog/python-date-comparison-comprehensive-tutorial/
                        return prediction # if less than 24 hours
        return None

    def save_predictions(self, fighter1, fighter2, prediction):
        """
        Saves or updates the prediction for the fight between fighter1 and fighter2.
        it saves if the tuple (fighter1, fighter2) is not in the database. If it is there,
        the method just updates the prediction and date. This avoids having duplicate rows
        and increasing the size of the database for out-dated predictions.
        """
        current_time = datetime.now().isoformat()  # source : https://www.geeksforgeeks.org/python/isoformat-method-of-datetime-class-in-python/
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO predictions (fighter1, fighter2, prediction, last_updated)
                VALUES (?, ?, ?, ?)
            """, (fighter1, fighter2, prediction, current_time))
            conn.commit()