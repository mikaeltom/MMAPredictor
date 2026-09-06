import os
import requests
from .AbstractAgent import AbstractAgent


class BetAgent(AbstractAgent):
    """
    BetAgent represents an agent that is responsible for
    fetching and observing the bookmakers opinion about
    futur fighter. It retrieves the odds using the API
    https://the-odds-api.com
    """
    def __init__(self, name):
        """
        Initializes the BetAgent with its api key and the
        url to get the bookmakers opinion (using the API).
        """
        super().__init__(name)
        self.api_key = os.environ["BET_TOKEN"]
        self.base_url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds" # source : https://the-odds-api.com

    def get_mma_odds(self, fighter1, fighter2):
        """
        Method that receives two fighters full name in parameters
        and try to find the odds related to the fight between them.
        First, it retrieves all events, and find the one with
        between both fighters. It uses parse_odd method to parse
        and get the odds related to the fight. Then return that.
        """
        params = { # source : https://the-odds-api.com
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        try:
            response = requests.get(self.base_url, params=params) # source: https://requests.readthedocs.io/en/latest/
        except Exception:
            return {"error": f"Error in request"}

        if response.status_code != 200: # source : https://stackoverflow.com/questions/54087303/how-to-check-for-200-ok
            return {"error": f"API error"} # no 200 so something was wrong, so error

        data = response.json() # source : https://www.geeksforgeeks.org/python/response-json-python-requests/
        if not data:
            return {"error": "No available MMA fights"}

        for event in data: # find the fights between the two fighters in events
            home = event["home_team"].lower() # lower to be sure correspondance with fighters
            away = event["away_team"].lower()
            f1 = fighter1.lower()
            f2 = fighter2.lower()
            if (f1 in home or f1 in away) and (f2 in home or f2 in away): # either (fighter1, fighter2) or (fighter2, fighter1)
                return self.parse_odds(event, fighter1, fighter2)

        return {
            "error": f"Fight '{fighter1} vs {fighter2}' not found",
        }

    def parse_odds(self, event, fighter1, fighter2):
        """
        Methods that receives event information about a fight
        (from the API), and the names of the two fighters.
        It returns the average of odds between the two fighters
        and other information such as the event date, the bookmakers
        data, the amount of bookmakers in a dictionary. LLMs understand
        better such format, that's why we did like this.
        """
        bookmakers_data = []
        for bookmaker in event["bookmakers"]: # source : https://the-odds-api.com
            for market in bookmaker["markets"]:
                if market["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]} # find the all the odds
                    bookmakers_data.append({ # source : https://the-odds-api.com
                        "bookmaker": bookmaker["title"],
                        "odds": outcomes,
                    })

        if not bookmakers_data: # in case something went wrong
            return {"error": "No fight found"}

        all_odds_f1 = [] # contains all the odds of the fighter
        all_odds_f2 = []
        for bk in bookmakers_data:
            for name, odd in bk["odds"].items():
                if fighter1.lower() in name.lower():
                    all_odds_f1.append(odd)
                elif fighter2.lower() in name.lower():
                    all_odds_f2.append(odd)

        avg_odds_f1 = sum(all_odds_f1) / len(all_odds_f1) if all_odds_f1 else None # take the average of the odds
        avg_odds_f2 = sum(all_odds_f2) / len(all_odds_f2) if all_odds_f2 else None

        return {
            "fight": f"{event['home_team']} vs {event['away_team']}",
            "commence_time": event["commence_time"],  # source : https://the-odds-api.com
            "fighter1": fighter1,
            "fighter2": fighter2,
            "avg_odds_fighter1": round(avg_odds_f1, 3) if avg_odds_f1 else None,
            "avg_odds_fighter2": round(avg_odds_f2, 3) if avg_odds_f2 else None,
            "bookmakers": bookmakers_data,
            "nb_bookmakers": len(bookmakers_data),
        }

    def process(self, *args):
        """
        Principale methods that handles the request received
        by the agent. The agent will retrieve the odds for the two
        fighters and return a readable summary.
        """
        fighter1 = str(args[0])
        fighter2 = str(args[1])
        return f" The bookmakers are giving the following odds : {self.get_mma_odds(fighter1, fighter2)}"