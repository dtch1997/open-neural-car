from abc import ABC, abstractmethod
from pathlib import Path


class BaseAgent(ABC):
    """Interface for an agent."""

    @abstractmethod
    def get_action(self, state):
        pass

    @staticmethod
    def load(self, save_path: Path) -> "BaseAgent":
        pass

    def save(self, save_path: Path):
        pass
