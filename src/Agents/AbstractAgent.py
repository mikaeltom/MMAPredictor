from abc import ABC, abstractmethod

class AbstractAgent(ABC):
    """
    Abstract Class that acts as a base for defining agents.
    """
    def __init__(self, name: str):
        """
        Initialize the agent with a name
        """
        self.name = name

    def get_name(self):
        """
        Returns the name of the agent.
        """
        return self.name

    @abstractmethod
    def process(self, *args):
        """
        Abstract method that every inherent agent
        will implement in order to processes the requests
        and communicate. It executes the agent logic
        """
        pass