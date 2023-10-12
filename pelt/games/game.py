from abc import ABC, abstractmethod
from typing import List, Tuple
from itertools import compress


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

    def play(self, model, adapters, tokenizer, config):
        # TODO: Consider replacing Game with play for game-specific custom logic.
        batch_size = config["batch_size"]
        history = [[]] * batch_size
        evals = [()] * batch_size
        active_timelines_mask = [True] * batch_size

        while any(active_timelines_mask):
            current_player = (len(history[0]) - 1) % 2
            current_adapter = adapters[current_player]
            current_model = model.set_adapter(current_adapter)

            # Only act in active timelines.
            active_timelines = list(compress(history, active_timelines_mask))

            # Keep track of stepped timelines so as to know how to put them back among others.
            active_timeline_idx = list(
                compress(range(len(active_timelines_mask)), active_timelines_mask)
            )

            # Step through active timelines and add them back.
            recent_timelines = self.act(current_model, tokenizer, active_timelines)
            recent_evals = self.eval(recent_timelines)
            for id, recent_timeline, recent_eval in zip(
                active_timeline_idx, recent_timelines, recent_evals
            ):
                history[id] = recent_timeline
                evals[id] = recent_eval

            # Active timelines are those where no non-zero scores have yet been assigned.
            active_timelines_mask = [not any(e) for e in evals]

        return history, evals
