import sqlite3
from datetime import datetime, timezone
import pandas as pd

def verify_if_fighter_exists(fighter):
    """
    Verifies if a given fighter stats exists in the database.
    """
    df1 = pd.read_csv("Data/fighter_recent.csv")
    df2 = pd.read_csv("Data/fighter_stats.csv")
    all_names = set(df1["name"].str.strip().str.lower())
    all_names = all_names.union(set(df2["name"].str.strip().str.lower()))
    return fighter.strip().lower() in all_names


def retrieve_checked_schedule():
    """
    Function to retrieve the checked schedule of fights from the database.
    The checked schedule means that the fighter stats are present in the
    database containing all the stats. So, we can process this fight. It also
    removes the duplicate fighters (fighters that combat multiple times on
    the day because some data bug of the API).
    """
    db_path = "Data/futurefights.db"
    now = datetime.now(timezone.utc) # source : https://www.geeksforgeeks.org/python/get-utc-timestamp-in-python/
    now = now.strftime("%Y-%m-%dT%H:%M:%SZ") # source : https://www.geeksforgeeks.org/python/python-strftime-function/
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT * FROM future_fights WHERE fight_date > ? ORDER BY fight_date ASC",
        (now,)
    ).fetchall()
    conn.close()
    lines = []
    seen_by_date = dict() # check to avoid having duplicates fighter on the same day (avoid bug of the api)
    for fights in rows:
        id, fighter1, fighter2, date = fights[0], str(fights[1]), str(fights[2]), fights[3]
        if verify_if_fighter_exists(fighter1) and verify_if_fighter_exists(fighter2): # check if exists
            if date not in seen_by_date:
                seen_by_date[date] = []
            if fighter1 not in seen_by_date[date] and fighter2 not in seen_by_date[date]: # check to not have duplicates on same day
                line = {"fighter1": fighter1, "fighter2": fighter2, "date": date}
                lines.append(line)
                seen_by_date[date].append(fighter1)
                seen_by_date[date].append(fighter2)
    return lines
