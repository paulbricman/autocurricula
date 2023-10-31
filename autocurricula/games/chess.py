from pettingzoo.classic import chess_v6

from typing import Dict, List


def preprocess(history: List[List[Dict]]) -> List[str]:
    """
    Helper function to put together prompt header for Chess.
    It involves rewinding the game for each timeline, string templating.
    """

    def preprocess_timeline(timeline):
        # For each timeline, determine the latest context.
        # History was previously `eval`-ed, so all legal.
        current_player = (len(timeline) - 1) % 2
        action_strings = [step["action"] for step in timeline]
        action_ints = iter([int(action) for action in action_strings])
        env = chess_v6.env(render_mode="ansi")
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

            env.step(action)

        header = (
            description()
            + "It is now your turn. First, reflect on the current state of the board:\n\n"
            + env.render()
            + "\n\nNext, carefuly think the situation through and reason about it:"
            " "
        )
        return header

    return [preprocess_timeline(timeline) for timeline in history]


def description() -> str:
    # From Wikipedia and PettingZoo game page:
    return """Chess is one of the oldest studied games in AI. Our implementation of the observation and action spaces for chess are what the AlphaZero method uses, with two small changes.

### Action Space

From the AlphaZero chess paper:

> [In AlphaChessZero, the] action space is a 8x8x73 dimensional array. Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or rook respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.

We instead flatten this into 8×8×73 = 4672 discrete action space.

You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a // (8*73), (a // 73) % 8, a % (8*73) % 73)`

Example:
    >>> x = 6
    >>> y = 0
    >>> c = 12
    >>> a = x*(8*73) + y*73 + c
    >>> print(a // (8*73), a % (8*73) // 73, a % (8*73) % 73)
    6 0 12

Note: the coordinates (6, 0, 12) correspond to column 6, row 0, plane 12. In chess notation, this would signify square G1:

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| A | B | C | D | E | F | G | H |

"""
