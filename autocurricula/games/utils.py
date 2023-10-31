from trl import AutoModelForCausalLMWithValueHead

from typing import List, Dict, Tuple
import math
import json


def set_player(model: AutoModelForCausalLMWithValueHead, player: Dict):
    if isinstance(player, dict):
        player = json.dumps(player)

    model.pretrained_model.set_adapter(player)


def action_ints_to_history(history: List[List[Dict]]):
    return [[{"action": action} for action in timeline] for timeline in history]


def is_integer(n) -> bool:
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def is_prime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if (n % i) == 0:
            return False
    return True


def illegal(illegal_player: int, num_players: int = 2) -> Tuple:
    scores = [None for _ in range(num_players)]
    scores[illegal_player] = -math.inf
    return tuple(scores)


def pz_eval(history: List[List[Dict]], env) -> List[Tuple]:
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
