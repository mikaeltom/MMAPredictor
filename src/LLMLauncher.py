import os
from openai import OpenAI
import json


class LLMLauncher(object):
    """
    Singleton service providing a centralized interface for LLM call. We used
    a singleton object to avoid having multiple class loading the model
    and having a lot of duplicate codes. We decided to use gpt-4o-mini
    as it is one of the free model on GitHub that has the most calls per day
    for free tier users. The singleton pattern is inspired by
    source : # source : https://python-patterns.guide/gang-of-four/singleton/
    """
    _instance = None
    client:OpenAI
    model:str

    def __init__(self):
        """
        Singleton pattern cannot be initialized directly.
        """
        raise RuntimeError('Call instance() instead') # source : https://python-patterns.guide/gang-of-four/singleton/

    @classmethod # source : https://python-patterns.guide/gang-of-four/singleton/
    def instance(cls, model="openai/gpt-4o-mini"):
        """
        For the Singleton pattern. Returns the singleton instance,
        and initialize openAI client if the singleton instance
        was not previously created.
        """
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            token = os.environ["GITHUB_TOKEN"]
            endpoint = "https://models.github.ai/inference"
            cls._instance.model = model
            cls._instance.client = OpenAI(base_url=endpoint, api_key=token, )
        return cls._instance

    def ask_LLM(self, system_prompt, messages_history):
        """
        Sends a request to the LLM API and returns the response as a json. This
        ensures that the output is consistent and the LLM do not responses in
        multiple different formats. The call combines the messages_history so that
        the LLM has context (this messages history might be empty  list if not needed).
        The following code is inspired by
        source : https://github.com/marketplace/models/azure-openai/gpt-4o/playground
        """
        messages = [{"role": "system", "content": system_prompt}] + messages_history
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"}
            )
            answer = response.choices[0].message.content
            result = json.loads(answer)
            return result
        except Exception as e:
            print("Error with LLM :", e)
            return None
