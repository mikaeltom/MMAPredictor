from .AbstractAgent import AbstractAgent
import prompt


class ContextAgent(AbstractAgent):
    """
    Context Agent is an agent that will receive the user input
    in the chat box and try to extract the two fighter names that
    the user was talking about. The idea is to avoid having too
    much LLM API calls (we have a limited amount of calls
    (free openAI plan) and it is not good for our planet). So,
    we have added a strategy of keyword extraction using Spacy.
    It tries to extract the two fighter names from the user prompt
    using Proper Nouns or Person (person feature of spacy). If it
    could not find two fighters using that or the two found names
    are not fighter names in the database, it uses an LLM to process
    the user input and extract the two fighter names. We added
    a mechanism against forms of prompt injection by VERIFYING
    the LLM output before returning it. The LLM extracts the two names,
    if these are not names from the fighter database (in
    form of dictionary), it returns an error message. Also, the LLM
    can answer negatively if the user input is not related to the
    prediction of fights (we did that in the prompt). This makes our
    LLM chatbot specific to prediction fights and nothing else, as seen
    in lecture 2 with Maxime Favyer.
    """
    def __init__(self, name, launcher, spacy_model):
        """
        Initializes the Context Agent. It takes a launcher object, that is a class
        containing the code necessary for communication with the oepnAI API. This allows
        to have less code duplication. It also has the spacy model, used for name
        extraction.
        """
        super().__init__(name)
        self.launcher = launcher
        self.nlp = spacy_model
        self.system_prompt = prompt.EXTRACTION_PROMPT
        self.fights = None

    def verified_fighter(self, fighter1, fighter2):
        """
        Method that checks if the user entered the correct fighter,
        and their fight is present in our database.
        """
        f1 = fighter1.strip().lower()
        f2 = fighter2.strip().lower()
        for fight in self.fights:
            if ((fight["fighter1"].lower() == f1 and fight["fighter2"].lower() == f2)
                    or (fight["fighter1"].lower() == f2 and fight["fighter2"].lower() == f1)): # both(f1 vs f2) or (f2 vs f1)
                return True
        return False

    def extract_nouns_and_persons(self, text):
        """
        Method that uses spacy to extract the two names of the fighters.
        It is inspired by spacy documentation :
        source: https://spacy.io/usage/linguistic-features
        """
        text = self.nlp(text)
        persons = []
        proper_nouns = []
        for token in text:
            if token.ent_type_ == "PERSON":
                persons.append(token.text)
            elif token.pos_ == "PROPN":
                proper_nouns.append(token.text)
        return persons, proper_nouns

    def extract_upper_letters(self, text):
        """
        Method that extract upper letters in python and test
        if the combination of these letters are give two
        valide fighters or not
        """
        text = text.split()
        upper_letter_words = [w for w in text if w[0].isupper()] # source : https://www.w3schools.com/PYTHON/ref_string_isupper.asp
        size = len(upper_letter_words)
        first_fighter_name = None
        second_fighter_name = None
        if size > 0:
            for i in range(size-1): # kind of sliding window testing consecutive pairs of names
                fighter = ""
                fighter += upper_letter_words[i]
                fighter += " " + upper_letter_words[i+1]
                fighter = fighter.strip().lower()
                for fight in self.fights: # try to find if the names are in one of the fighters
                    if fight["fighter1"].lower() == fighter:
                        first_fighter_name = fight["fighter1"]
                    elif fight["fighter2"].lower() == fighter:
                        second_fighter_name = fight["fighter2"]
        if ((first_fighter_name is not None and second_fighter_name is not None)
                and self.verified_fighter(first_fighter_name, second_fighter_name)): # verify if fighting against each other
            return True, first_fighter_name, second_fighter_name
        return False, None, None

    def compose_person_name(self, persons, proper_nouns):
        """
        Method that composes the person name from the proper nouns. Spacy
        feature person do not have every fighter, so we try combine the Proper Nouns
        and/or Persons that were found to create two fighters names.
        """
        if len(persons) == 2: # two person
            return persons[0], persons[1]
        elif len(proper_nouns) == 4: # 4 propres nouns
            return proper_nouns[0] + " " + proper_nouns[1], proper_nouns[2] + " " + proper_nouns[3]
        elif len(persons) == 1 and len(proper_nouns) == 2: # one person and 2 proper nouns
            return persons[0], proper_nouns[0] + " " + proper_nouns[1]
        return None, None

    def process(self, *args):
        """
        Method that processes the user input and extract the two fighter names.
        It tries to extract the two fighter names from the user prompt using Spacy
        to not use too much LLM calls. If it can not find two fighters names using that
        technique, it tries to extract the two fighter names from the user prompt using an
        LLM. The LLM result is re-checked to avoid hallucinations and other kind of security
        problems such as prompt injection (LLM can not communicate other things that name
        so in case of prompt injection, the output will not be fighter names, so it is discard).
        """
        request = args[0]
        messages_history = args[1]
        self.fights = args[2] # list of future fights

        is_upper_letter, fighter_1, fighter_2 = self.extract_upper_letters(request)

        if not is_upper_letter:
            extracted_nouns_and_persons = self.extract_nouns_and_persons(request)
            fighter_1, fighter_2 = self.compose_person_name(extracted_nouns_and_persons[0], extracted_nouns_and_persons[1])

        if fighter_1 is None or fighter_2 is None or not (self.verified_fighter(fighter_1, fighter_2)): # ask LLM if not found fighters
            result_json = self.launcher.ask_LLM(self.system_prompt + request + f" Here are the scheduled fights : {self.fights}",
                [{"role": "user","content": f"Here is the message history, look at it only if you do not find names in the original user query: {messages_history}"}])
            if result_json is None:
                return False, "[Error] I am busy, please try again later..."

            action = result_json["action"]
            if action == "other": # if the LLM return the keycode 'other' it means that it was asked something else than fight
                return False, "[Error] : This is not my job ! I am an LLM that can only predict real MMA fights."

            fighters = result_json["fighters"]
            fighter_1 = fighters[0]['first_name'] + " " + fighters[0]['last_name']
            fighter_2 = fighters[1]['first_name'] + " " + fighters[1]['last_name']

            if not (self.verified_fighter(fighter_1, fighter_2)): # verify LLM output, anti-hallucination (or injection)
                return False, "[Error] : The proposed fight does not exist ! Please try again."

        return True, [fighter_1, fighter_2]