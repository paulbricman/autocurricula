from typing import List, Tuple


def action_ints_to_history(
    history: List[List[int]],
) -> List[List[Tuple[str, str, str]]]:
    return [[("", "", str(action)) for action in timeline] for timeline in history]


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def illegal(illegal_player, num_players=2):
    scores = [0] * num_players
    scores[illegal_player] = -1
    return tuple(scores)
