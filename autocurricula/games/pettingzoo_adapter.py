from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from typing import List, Dict, Tuple
import math
import json


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
