from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    """Interface for an agent."""

    @abstractmethod
    def get_action(self, state):
        pass

    def reset(self, env):
        pass
