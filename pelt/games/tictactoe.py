from pelt.games.utils import is_integer, illegal
from typing import List, Tuple
from pettingzoo.classic import tictactoe_v3
import numpy as np
from itertools import compress
import json


def play(model, match, tokenizer, config):
    """
    Given a list of players and a tokenizer, return the outcomes of the game (i.e. evals, history).

    Args:
        model: peft-wrapped model backbone
        match: list of player dicts
        tokenizer: `transformers` tokenizer used by model
        config: standard config dict ("game" subdict being most relevant)

    Returns:
        Tuple of evals and play history.
    """
    batch_size = config["game"]["batch_size"]
    history = [[] for _ in range(batch_size)]
    evals = [() for _ in range(batch_size)]
    active_timelines_mask = [True for _ in range(batch_size)]

    while any(active_timelines_mask):
        current_player_id = len(history[0]) % 2
        current_player = match[current_player_id]

        if isinstance(current_player, dict):
            current_player = json.dumps(current_player)

        model.pretrained_model.set_adapter(current_player)

        # Only act in active timelines.
        active_timelines = list(compress(history, active_timelines_mask))

        # Keep track of stepped timelines so as to know how to put them back among others.
        active_timeline_idx = list(
            compress(range(len(active_timelines_mask)), active_timelines_mask)
        )

        # Step through active timelines and add them back.
        recent_timelines = act(model, tokenizer, active_timelines)
        for id, recent_timeline in enumerate(recent_timelines):
            active_timelines[id] += [recent_timeline]

        recent_evals = eval(active_timelines)
        for id, recent_eval in zip(active_timeline_idx, recent_evals):
            evals[id] = recent_eval

        # Active timelines are those where no non-zero scores have yet been assigned.
        active_timelines_mask = [not any(e) for e in evals]

    return evals, history


def act(model, tokenizer, history):
    """
    Given the history of the game, a model, and a tokenizer, produce an intermediate reasoning trace and an action.

    Args:
        model: `transformers` or `peft`-wrapped model
        tokenizer: `transformers` tokenizer used by model
        history: past actions and associated trains of thought (B x [T x (E x [(context, thought)], action)]).

    Returns:
        Yet another ([(context, thought), (extended_context, action)], action) object.
    """
    # First, work towards generating a reasoning trace.
    contexts = preprocess(history)
    contexts_ids = tokenizer(contexts, return_tensors="pt")

    thoughts_ids = model.generate(
        **contexts_ids,
        min_new_tokens=10,
        max_new_tokens=20,
        suppress_tokens=[198, 628],
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
        max_new_tokens=3,
        suppress_tokens=[198, 628],
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

    actions = [
        {
            "thoughts": [
                {
                    "context": e[0][0],
                    "behavior": e[0][1],
                },
                {
                    "context": e[1][0],
                    "behavior": e[1][1],
                },
            ],
            "action": e[1][1],
        }
        for e in context_actions
    ]

    return actions


def preprocess(history):
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


def eval(history):
    """
    Given play history, compute player evals.
    """

    def eval_timeline(timeline):
        last_move_player = (len(timeline) - 1) % 2
        action_strings = [step["action"] for step in timeline]

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
                    return illegal(last_move_player)
            return tuple(env.rewards.values())
        else:
            return illegal(last_move_player)

    return [eval_timeline(timeline) for timeline in history]


def description():
    # From PettingZoo game page:
    return """Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:

0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8

"""
