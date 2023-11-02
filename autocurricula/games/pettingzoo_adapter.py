from autocurricula.games.utils import set_player

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from typing import List, Dict, Tuple
from itertools import compress
import math
import json
import os


def play(
    match: Tuple[Dict],
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    preprocess,
    env,
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

        recent_evals = eval(active_timelines, env)
        for id, recent_eval in zip(active_timeline_idx, recent_evals):
            evals[id] = recent_eval

        # Active timelines are those where no non-zero scores have yet been assigned.
        active_timelines_mask = [not any(e) for e in evals]

    return evals, history


def act(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    history: List[List[Dict]],
    preprocess,
) -> List[Dict]:
    """
    Given the history of the game, a model, and a tokenizer, produce an intermediate
    reasoning trace and an action.
    """
    # First, work towards generating a reasoning trace.
    contexts = preprocess(history)
    contexts_ids = tokenizer(contexts, return_tensors="pt")

    if os.environ.get("PJRT_DEVICE") == "TPU":
        import torch_xla.core.xla_model as xm

        contexts_ids.to(xm.xla_device())

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

    if os.environ.get("PJRT_DEVICE") == "TPU":
        extended_contexts_ids.to(xm.xla_device())

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


def eval(history: List[List[Dict]], env) -> List[Tuple]:
    """
    Dedicated PettingZoo eval wrapper because of boilerplate.
    Given play history, compute player evals.
    """

    def eval_timeline(timeline):
        last_move_player = (len(timeline) - 1) % 2
        action_strings = [step["action"] for step in timeline]

        # Even before game-legal moves, we need to have PZ-compatible moves.
        if is_integer(action_strings[-1]):
            # If all good, rewind PZ env using recorded actions.
            action_ints = iter([int(action) for action in action_strings])
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

                # Apply action, unless game-illegal, case in which end game.
                try:
                    env.step(action)
                except AssertionError:
                    return illegal(last_move_player)

                # If we've got a winner, return rewards.
                rewards = tuple(env.rewards.values())
                if any([reward != 0 for reward in rewards]):
                    return rewards
            return tuple(env.rewards.values())
        else:
            return illegal(last_move_player)

    return [eval_timeline(timeline) for timeline in history]


def is_integer(n) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def illegal(illegal_player: int, num_players: int = 2) -> Tuple:
    scores = [None for _ in range(num_players)]
    scores[illegal_player] = -math.inf
    return tuple(scores)
