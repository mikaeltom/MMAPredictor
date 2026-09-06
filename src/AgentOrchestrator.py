from Agents.BetAgent import BetAgent
from Agents.NewsAgent import NewsAgent
from Agents.PredictorAgent import PredictorAgent
from Agents.AbstractAgent import AbstractAgent
from Agents.LLMExplainerAgent import LLMExplainerAgent
from Agents.FutureFightFinderAgent import FutureFightFinderAgent
from Agents.HistoricalAgent import HistoricalAgent
from Agents.ContextAgent import ContextAgent
from LLMLauncher import LLMLauncher
import concurrent.futures
from PredictionCache import PredictionCache


class AgentOrchestrator:
    """
    This class acts as the orchestrator for an agent. As seen in lecture 2 of
    Maxime Favyer, when he speaks about orchestrator. Here this class is the
    central controller of the agents. It manages their life cycle, handles
    request of fights, give the agent tasks in parallel, and send the different
    agent analysis to the LLM explainer, to guide its response/prediction. The
    Agent Orchestrator uses a prediction cache mechanism to avoid calling too
    much the LLM and other API (as we have a limited amount of requests). The idea
    is verified if the prediction of a fight, given the two fighters name, was already
    given less than 24 hours ago, if it is the case, it just gives this prediction back,
    without having to call the LLM and all other APIs. In contrast, if the prediction
    was given more than 24 hours ago, it processes with the full prediction pipeline and
    save the answer to the cache (a database sqlite3).
    """
    def __init__(self, spacy_model):
        """
        Initialize the orchestrator, setups the agents (futurefighter, bet, historcial,
        news, predictor, context, llm explainer), the cache, the LLM launcher, a centralized
        component that handles the LLM calls in a single place to avoid code duplication.
        """
        self.cache = PredictionCache()
        self.llm_launcher = LLMLauncher.instance()
        self.future_fights_finder = FutureFightFinderAgent("FutureFightsFinder").process() # retrieve calendar of fights
        self.list_agents : list[AbstractAgent] = [BetAgent("Bet"), NewsAgent("News", self.llm_launcher), PredictorAgent("Predictor"), HistoricalAgent("Historical")]
        self.llm_explainer_agent = LLMExplainerAgent("LLMComparator", self.llm_launcher)
        self.context_agent = ContextAgent("Context", launcher=self.llm_launcher, spacy_model=spacy_model)
        self.messages_history = [] # contains all the messages send by the user during this session, allows having more context to the LLM

    def process_request(self, request, fights):
        """
        Handles the ChatBox request of the user. It delagates the handling
        to the Context Agent, that returns the names of the fighters, then uses
        these two names in the prediction pipeline.
        """
        validity, description = self.context_agent.process(request, self.messages_history, fights)
        if validity:
            fighter_1, fighter_2 = description[0], description[1]
            self.messages_history.append({"role": "user", "content": request})
            response = self.predict_winner(fighter_1, fighter_2)
            return response
        else:
            return description

    def run_agent(self, agent, fighter_1, fighter_2):
        """
        Method used for the multi-threading purpose.
        It executes the given agent process method.
        """
        agent_id = agent.get_name()
        agent_communication = agent.process(fighter_1, fighter_2)
        return agent_id, agent_communication

    def predict_winner(self, fighter_1, fighter_2):
        """
        Orchestrate the prediction process. It checks the cache for
        existing results, execute all analysis using multi-threading to
        make the agent work in parallel. Aggregates the different agents
        analysis and gives that to the LLM explainer, that represent the
        guided LLM that makes the final decision using the analysis.
        """
        prediction = self.cache.retrieve_predictions(fighter_1, fighter_2)
        if prediction is not None:
            return prediction

        list_responses = dict() # contains the analysis of each agent with their name
        with concurrent.futures.ThreadPoolExecutor() as executor: # source : https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3
            futures = []
            for agent in self.list_agents:
                futures.append(executor.submit(self.run_agent, agent, fighter_1, fighter_2))

            for future in concurrent.futures.as_completed(futures): # allow multiple agent to perform different task in parallel
                result = future.result()
                list_responses[result[0]] = result[1]

        validity, prediction = self.llm_explainer_agent.process(fighter_1, fighter_2, list_responses)
        if not validity: # if LLM had a problem
            return "Something went wrong. Please try again later."
        self.cache.save_predictions(fighter_1, fighter_2, prediction) # save new prediction in cache
        return prediction
