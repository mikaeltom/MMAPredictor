import os
import requests
from Agents.AbstractAgent import AbstractAgent
import prompt


class NewsAgent(AbstractAgent):
    """
    News Agent is and agent that retrieves recent
    information about the two fighters. The idea is to use
    Tavily API : https://www.tavily.com to search for
    news about the two fighters and summarize them using an LLM.
    Our intuition is that psychology and wellness of the fighters
    play an important role in the outcome of the fight. If they have
    some small injuries or other bad news, it can completely impact
    their performances.
    """
    def __init__(self, name, launcher):
        """
        Initializes the News Agent with its required API keys and the
        centralized project LLMLauncher in order to avoid code duplication.
        """
        super().__init__(name)
        self.tavily_key = os.environ.get("TAVILY_API_KEY")
        self.llm_launcher = launcher
        self.url = "https://api.tavily.com/search"
        self.prompt = prompt.NEWS_AGENT_PROMPT

    def search_web(self, query):
        """
        Executes a web search using Tavily API to get raw text content
        from MMA news websites. Uses the Tavily search engine to target
        specific keywords (injuries, training, weight cuts) to ensure
        the gathered data is contextually valuable for fight prediction analysis.
        """
        if not self.tavily_key:
            print(f"TAVILY_API_KEY is missing from environment variables.")
            return []

        url = self.url
        payload = { # source : https://docs.tavily.com/examples/quick-tutorials/search-api
            "api_key": self.tavily_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False, # no own generated summary by tavily
            "max_results": 5 # top 5 best article for our system
        }
        try:
            response = requests.post(url, json=payload) # source : https://stackoverflow.com/questions/38847317/sending-payload-for-get-request-in-python
            if response.status_code == 200: # source : https://stackoverflow.com/questions/54087303/how-to-check-for-200-ok
                return response.json().get("results", []) # source : https://www.geeksforgeeks.org/python/response-json-python-requests/
            else:
                print(f"Tavily API error")
                return []
        except Exception:
            print(f"[Tavily Failed during web search")
            return []

    def analyze_news(self, fighter1, fighter2, raw_news):
        """
        Leverages LLMLauncher to parse raw news snippets
        and extract structured JSON analytical data.
        """
        if not raw_news:
            return {
                "has_critical_info": False, # critical info means that there is something important (if false, nothing important juste promotion)
                "summary": f"No recent significant news found for {fighter1} vs {fighter2}."
            }
        context = ""
        for idx, news in enumerate(raw_news): # for each news, we get the article and the title and the news, better organization for LLM
            context += f"- Source {idx + 1}: {news['title']} ---\n{news['content']}\n\n"

        system_prompt = self.prompt
        user_content = f"""
        Analyze the upcoming fight between **{fighter1}** and **{fighter2}** using these articles:
        {context}
        """
        messages_history = [{"role": "user", "content": user_content}]
        result = self.llm_launcher.ask_LLM(system_prompt, messages_history)

        if result is None:
            return {"has_critical_info": False, "summary": "Error during LLM analysis generation."}

        return result

    def process(self, *args):
        """
        Main entrypoint mandated by AbstractAgent.
        Expects args[0] to be fighter1 (str) and args[1] to be fighter2 (str).
        Returns a structured dictionary with the fight news analysis.
        """
        fighter1 = str(args[0])
        fighter2 = str(args[1])

        search_query = f"{fighter1} {fighter2} fight injury training weight cut camp" # search we are going to do

        raw_results = self.search_web(search_query)
        report_json = self.analyze_news(fighter1, fighter2, raw_results)

        return report_json