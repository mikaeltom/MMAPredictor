from .AbstractAgent import AbstractAgent
import sqlite3
import requests
from datetime import datetime, timedelta
import os


class FutureFightFinderAgent(AbstractAgent):
    """
    FutureFightFinderAgent is an agent that scrap the API : https://the-odds-api.com
    to retrieve futur fights that are planned. Its objective is to find
    the scheduled UFC fights and put these fights into a database sqlite3 with the date
    of the fight. It handles this databases modification with a specific mechanism to
    avoid too much API calls. The idea is that the futur fight database is updated
    every week. In the database, a table contain the last update date, if it less
    than 1 week old, we do not update the database. We estimated that 1 week is a
    fair update timing as UFC calendars are fixed months and year in advances,
    with not too much changes.
    """
    def __init__(self, name):
        """
        Initializes the FutureFightFinderAgent, sets up database paths, and
        prepares the connection to the API.
        """
        super().__init__(name)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_name = os.path.join(root, "Data", "futurefights.db")
        self.api_key = os.environ["BET_TOKEN"]
        self.base_url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
        self.init_db()

    def init_db(self):
        """
        Initializes the database connection and creates the tables. This mechanism is
        there to avoid errors related to table deletion. We check of the table exists or
        not, if not we create that. This ensures that the database is always present in our
        system. It creates two tables, the scheduled fights table and table with update time.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS future_fights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fighter1 TEXT,
            fighter2 TEXT,
            fight_date TEXT,
            UNIQUE(fighter1, fighter2, fight_date)
            )
            """)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            last_updated TEXT NOT NULL
            )
            """)
            conn.commit()

    def retrieve_fights(self):
        """
        Method that retrieves all the futur fight data from the API.
        It queries the API and return a list of the futur fights in
        dictionary format.
        """
        params = { # source : https://the-odds-api.com
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            response = requests.get(self.base_url, params=params) # source: https://requests.readthedocs.io/en/latest/
            if response.status_code != 200:  # source : https://stackoverflow.com/questions/54087303/how-to-check-for-200-ok
                print(f"API error")
                return []
            data = response.json() # source : https://www.geeksforgeeks.org/python/response-json-python-requests/
        except Exception:
            print(f"Failed to list fights")
            return []

        return [
            {
                "home": event["home_team"],
                "away": event["away_team"],
                "commence_time": event.get("commence_time"),
            } # source : https://the-odds-api.com
            for event in data
        ]

    def needs_update(self):
        """
        Method that verifies if the futur fight database needs to be updated.
        If the last update date is less than 1 week old, it will not update
        the database, to avoid too much API calls.
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT last_updated FROM update_log ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
        if row is None:
            return True
        last_date = row[0]
        last_date = datetime.fromisoformat(last_date) # source : https://www.geeksforgeeks.org/python/fromisoformat-function-of-datetime-date-class-in-python/
        return datetime.now() - last_date > timedelta(days=7) # source : https://www.influxdata.com/blog/python-date-comparison-comprehensive-tutorial/

    def process(self, *args):
        """
        Method that retrieve all the futur fight data from the API.
        If last update is less than 1 week old, it will not update the
        database (avoid too much API calls).
        """
        if not self.needs_update():
            return

        fights = self.retrieve_fights()
        if not fights:
            return
        current_time = datetime.now().isoformat() # source : https://www.geeksforgeeks.org/python/isoformat-method-of-datetime-class-in-python/
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.cursor()
            for fight in fights:
                cursor.execute("""
                INSERT OR IGNORE INTO future_fights (fighter1, fighter2, fight_date)
                VALUES (?, ?, ?)
                """, (fight["home"], fight["away"], fight["commence_time"])) # source : https://the-odds-api.com
            cursor.execute("""
            INSERT INTO update_log (last_updated)
            VALUES (?)
            """, (current_time,))
            conn.commit()
