from abc import ABC, abstractmethod
from typing import List, Tuple


class Game(ABC):
    @abstractmethod
    def act(
        self, model, tokenizer, history: List[List[Tuple[List[Tuple[str, str]], str]]]
    ):
        """
        You woke up from amnesia. You're in the middle of a game. `history` includes
        past actions. You think through the game step and eventually produce an action.

        Args:
            model: `transformers` or `peft`-wrapped model
            tokenizer: `transformers` tokenizer used by model
            history: past actions and associated trains of thought (B x [T x (E x [(context, thought)], action)]).

        Returns:
            Yet another (E x [context, thought], action) triple.
        """
        pass

    @abstractmethod
    def eval(self, history: List[List[Tuple[List[Tuple[str, str]], str]]]):
        """
        You woke up from amnesia. Players are playing a game, and you're the judge.
        `history` contains their actions. You determine whether the game should end, and if so, the players' scores, too.

        Args:
            history: past actions and associated trains of thought (B x [T x (E x [(context, thought)], action)]).

        Returns:
            If the game should continue, returns `None`. Otherwise, returns list of player scores.
        """
        pass
