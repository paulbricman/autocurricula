from pelt.games.game import Game
from pelt.games.utils import is_integer, illegal
from abc import ABC, abstractmethod
from typing import List, Tuple
from pettingzoo.classic import tictactoe_v3
import numpy as np


class TicTacToe(Game):
    def act(self, model, tokenizer, history):
        """
        Given the history of the game, and a model and tokenizer, use the former to
        prompt the latter two to produce an intermediate reasoning trace and an action.

        Args:
            model: `transformers` or `peft`-wrapped model
            tokenizer: `transformers` tokenizer used by model
            history: past actions and associated trains of thought (B x [T x (E x [(context, thought)], action)]).

        Returns:
            Yet another ([(context, thought), (extended_context, action)], action) triple.
        """
        # First, work towards generating a reasoning trace.
        contexts = self._preprocess(history)
        contexts_ids = tokenizer(contexts, return_tensors="pt")

        thoughts_ids = model.generate(
            **contexts_ids,
            min_new_tokens=10,
            max_new_tokens=20,
            suppress_tokens=[198, 628, 50256],
            do_sample=True,
        )
        thoughts = tokenizer.batch_decode(thoughts_ids)

        # Second, use the reasoning trace to work towards an action.
        extended_contexts = [
            e + "...\n\nFinally, provide your intended action:" for e in thoughts
        ]
        extended_contexts_ids = tokenizer(
            extended_contexts, return_tensors="pt", padding=True, truncation=True
        )

        action_ids = model.generate(
            **extended_contexts_ids,
            min_new_tokens=1,
            max_new_tokens=5,
            suppress_tokens=[198, 628, 50256],
            do_sample=True,
        )
        actions = tokenizer.batch_decode(action_ids)

        # Package things up nicely into experiences.
        trimmed_thoughts = []
        trimmed_actions = []
        for context, extended_context, thought, action in zip(
            contexts, extended_contexts, thoughts, actions
        ):
            trimmed_thoughts += [thought[len(context) :]]
            trimmed_actions += [action[len(extended_context) :]]

        context_actions = zip(
            zip(contexts, trimmed_thoughts), zip(extended_contexts, trimmed_actions)
        )

        experiences = list(zip(context_actions, trimmed_actions))
        return experiences

    def _preprocess(self, history: List[List[Tuple[List[Tuple[str, str]], str]]]):
        def preprocess_timeline(timeline):
            # For each timeline, determine the latest context.
            # History was previously `eval`-ed, so all legal.
            current_player = (len(timeline) - 1) % 2
            action_strings = [step[-1] for step in timeline]
            action_ints = iter([int(action) for action in action_strings])
            env = tictactoe_v3.env()
            env.reset()
            latest_obs = None

            for _ in env.agent_iter():
                latest_obs, _, termination, truncation, _ = env.last()
                if termination or truncation:
                    action = None
                else:
                    try:
                        action = next(action_ints)
                    except StopIteration:
                        break
                env.step(action)

            def obs_to_board_string(obs):
                # TicTacToe env lacks ansi render.
                symbols = ["X", "O"]
                obs = np.swapaxes(obs["observation"], 0, 1)

                board = ""
                for col_idx, col in enumerate(obs):
                    for row_idx, row in enumerate(col):
                        if row[0]:
                            board += symbols[current_player]
                        elif row[1]:
                            board += symbols[1 - current_player]
                        else:
                            board += "-"
                        if row_idx != 2:
                            board += " | "
                    if col_idx != 2:
                        board += "\n_________\n\n"
                return board

            header = (
                # self.description
                ""
                + "It is now your turn. First, reflect on the current state of the board:\n\n"
                + obs_to_board_string(latest_obs)
                + "\n\nNext, carefuly think the situation through and reason about it:"
            )
            return header

        return [preprocess_timeline(timeline) for timeline in history]

    def eval(self, history: List[List[Tuple[List[Tuple[str, str]], str]]]):
        def eval_timeline(timeline):
            current_player = (len(timeline) - 1) % 2
            action_strings = [step[-1] for step in timeline]

            # Even before game-legal moves, we need to have PZ-compatible moves.
            if is_integer(action_strings[-1]):
                # If all good, rewind PZ env using recorded actions.
                action_ints = iter([int(action) for action in action_strings])
                env = tictactoe_v3.env()
                env.reset()

                for _ in env.agent_iter():
                    _, _, termination, truncation, _ = env.last()
                    if termination or truncation:
                        action = None
                    else:
                        try:
                            action = next(action_ints)
                        except StopIteration:
                            break

                    # If we've got a winner, return rewards.
                    rewards = tuple(env.rewards.values())
                    if any([reward != 0 for reward in rewards]):
                        return rewards

                    # Apply action, unless game-illegal, case in which end game.
                    try:
                        env.step(action)
                    except AssertionError:
                        return illegal(current_player)
                return tuple(env.rewards.values())
            else:
                return illegal(current_player)

        return [eval_timeline(timeline) for timeline in history]

    # From PettingZoo game page:
    description = """Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:

0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8

"""
