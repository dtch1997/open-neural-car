from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Interface for an agent."""

    @abstractmethod
    def get_action(self, state):
        pass
