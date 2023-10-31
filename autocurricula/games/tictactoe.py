from autocurricula.games.utils import set_player
from autocurricula.games.pettingzoo_adapter import act, eval

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from pettingzoo.classic import tictactoe_v3

from typing import Tuple, Dict, List
from itertools import compress
import numpy as np
import json


def play(
    match: Tuple[Dict],
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
) -> Tuple[List[Tuple], List[List[Dict]]]:
    """
    Given a list of players and a tokenizer, return the outcomes of the
    game (i.e. evals, history).
    """
    batch_size = 4
    history = [[] for _ in range(batch_size)]
    evals = [() for _ in range(batch_size)]
    active_timelines_mask = [True for _ in range(batch_size)]

    while any(active_timelines_mask):
        current_player_id = len(history[0]) % 2
        current_player = match[current_player_id]

        if isinstance(current_player, dict):
            current_player = json.dumps(current_player)

        set_player(model, current_player)

        # Only act in active timelines.
        active_timelines = list(compress(history, active_timelines_mask))

        # Keep track of stepped timelines so as to know how to put them back among others.
        active_timeline_idx = list(
            compress(range(len(active_timelines_mask)), active_timelines_mask)
        )

        # Step through active timelines and add them back.
        recent_timelines = act(model, tokenizer, active_timelines, preprocess)
        for id, recent_timeline in enumerate(recent_timelines):
            active_timelines[id] += [recent_timeline]

        recent_evals = eval(active_timelines, tictactoe_v3.env())
        for id, recent_eval in zip(active_timeline_idx, recent_evals):
            evals[id] = recent_eval

        # Active timelines are those where no non-zero scores have yet been assigned.
        active_timelines_mask = [not any(e) for e in evals]

    return evals, history


def preprocess(history: List[List[Dict]]) -> List[str]:
    """
    Helper function to put together prompt header for TicTacToe.
    It involves rewinding the game for each timeline, string templating.
    """

    def preprocess_timeline(timeline):
        # For each timeline, determine the latest context.
        # History was previously `eval`-ed, so all legal.
        current_player = (len(timeline) - 1) % 2
        action_strings = [step["action"] for step in timeline]
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
            # TicTacToe env lacks native ansi render.
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
            description()
            + "It is now your turn. First, reflect on the current state of the board:\n\n"
            + obs_to_board_string(latest_obs)
            + "\n\nNext, carefuly think the situation through and reason about it:"
            " "
        )
        return header

    return [preprocess_timeline(timeline) for timeline in history]


def description() -> str:
    # From PettingZoo game page:
    return """Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:

0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8

"""
