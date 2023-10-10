from typing import List, Tuple


def action_ints_to_history(
    history: List[List[int]],
) -> List[List[Tuple[str, str, str]]]:
    return [[("", "", str(action)) for action in timeline] for timeline in history]
