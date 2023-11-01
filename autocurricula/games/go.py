from autocurricula.games.pettingzoo_adapter import play as pz_play

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from pettingzoo.classic import go_v5

from typing import Tuple, Dict, List
import numpy as np


board_size = 9


def play(
    match: Tuple[Dict],
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
) -> Tuple[List[Tuple], List[List[Dict]]]:
    """
    Given a list of players and a tokenizer, return the outcomes of the
    game (i.e. evals, history).
    """
    return pz_play(
        match, model, tokenizer, preprocess, go_v5.env(board_size=board_size)
    )


def preprocess(history: List[List[Dict]]) -> List[str]:
    """
    Helper function to put together prompt header for Go.
    It involves rewinding the game for each timeline, string templating.
    """

    def preprocess_timeline(timeline):
        # For each timeline, determine the latest context.
        # History was previously `eval`-ed, so all legal.
        current_player = (len(timeline) - 1) % 2
        action_strings = [step["action"] for step in timeline]
        action_ints = iter([int(action) for action in action_strings])
        env = go_v5.env(board_size=board_size)
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
            # Go env lacks native ansi render.
            symbols = ["B", "W"]
            obs = np.swapaxes(obs["observation"], 0, 1)

            board = ""
            for col_idx, col in enumerate(obs):
                for _, row in enumerate(col):
                    if row[0]:
                        board += symbols[current_player]
                    elif row[1]:
                        board += symbols[1 - current_player]
                    else:
                        board += "."
                    board += "\t"
                if col_idx != board_size - 1:
                    board += "\n"
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
    base = f"""Go is a board game with 2 players, black and white. The black player starts by placing a black stone at an empty board intersection. The white player follows by placing a stone of their own, aiming to either surround more territory than their opponent or capture the opponent’s stones. The game ends if both players sequentially decide to pass.

Each action from 0 to {board_size ** 2 - 1} represents placing either a white or a black stone in the corresponding cell. The cells are indexed as follows:

"""
    for row in range(board_size):
        for col in range(board_size):
            base += str(col + row * board_size) + "\t"
            if col == board_size - 1:
                base += "\n"

    base += f"\nAction {board_size ** 2} represents a pass.\n\n"
    return base
