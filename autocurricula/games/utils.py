from trl import AutoModelForCausalLMWithValueHead

from typing import List, Dict, Tuple


def set_player(model: AutoModelForCausalLMWithValueHead, player: Dict):
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


def illegal(illegal_player: int, num_players: int = 2) -> Tuple:
    scores = [0] * num_players
    scores[illegal_player] = -1
    return tuple(scores)
