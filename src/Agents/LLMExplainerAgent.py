from .AbstractAgent import AbstractAgent
import prompt


class LLMExplainerAgent(AbstractAgent):
    """
    Agent that receives the analysis of all the other agents. This agent
    is the decision maker that uses the given information to predict who
    is going to be the winner of the asker fight. It uses
    an LLM to predict the winner of the asker fight. This is
    the "guided" LLM by the agent analysis in our system.
    """
    def __init__(self, name, launcher):
        """
        Initializes the Explainer Agent with an LLM launcher and the system prompt template.
        """
        super().__init__(name)
        self.launcher = launcher
        self.system_prompt = prompt.PREDICTION_PROMPT

    def generate_sentence(self, text_json):
        """
        Formats the structured JSON response from the LLM into a readable textual explanation.
        """
        winner = text_json["winner"]
        method = text_json["method"]
        confidence = text_json["confidence"]
        explanation = text_json["explanation"]
        result = f"{winner} will win by {method}. Confidence is {confidence}.\n{explanation}"
        return result

    def process(self, *args):
        """
        Main execution method. Aggregates matchup information and external context
        to request a comprehensive fight analysis and prediction from the LLM.
        """
        fighter1 = str(args[0])
        fighter2 = str(args[1])
        list_responses = args[2]

        prompt_text = (self.system_prompt + "The fight is between first fighter:" + fighter1
                       + " vs second fighter:" + fighter2 + ". " + str(list_responses))
        result_json = self.launcher.ask_LLM(prompt_text, [])

        if result_json is None:
            return False, "[Error] I am busy, please try again later..."

        return True, self.generate_sentence(result_json)