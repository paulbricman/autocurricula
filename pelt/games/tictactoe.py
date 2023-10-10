from pelt.games.game import Game
from abc import ABC, abstractmethod
from typing import List, Tuple
from pettingzoo.classic import tictactoe_v3


class TicTacToe(Game):
    def act(self, history: List[List[Tuple[str, str, str]]]):
        pass

    def eval(self, history: List[List[Tuple[str, str, str]]]):
        def is_integer(n):
            try:
                float(n)
            except ValueError:
                return False
            else:
                return float(n).is_integer()

        def illegal(is_player_one):
            if is_player_one:
                return (-1, 0)
            else:
                return (0, -1)

        def eval_timeline(timeline):
            is_player_one = len(timeline) % 2 == 1
            action_strings = [step[-1] for step in timeline]

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
                        action = next(action_ints)

                    rewards = tuple(env.rewards.values())
                    if any([reward != 0 for reward in rewards]):
                        return rewards

                    try:
                        env.step(action)
                    except AssertionError:
                        return illegal(is_player_one)

                print(env)
                return tuple(env.rewards.values())
            else:
                return illegal(is_player_one)

        rewards = [eval_timeline(timeline) for timeline in history]
        return rewards

    description = """
# Tic Tac Toe

```{figure} classic_tictactoe.gif
:width: 140px
:name: tictactoe
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic import tictactoe_v3` |
|--------------------|-----------------------------------------------|
| Actions            | Discrete                                      |
| Parallel API       | Yes                                           |
| Manual Control     | No                                            |
| Agents             | `agents= ['player_1', 'player_2']`            |
| Agents             | 2                                             |
| Action Shape       | (1)                                           |
| Action Values      | [0, 8]                                        |
| Observation Shape  | (3, 3, 2)                                     |
| Observation Values | [0,1]                                         |


Tic-tac-toe is a simple turn based strategy game where 2 players, X and O, take turns marking spaces on a 3 x 3 grid. The first player to place 3 of their marks in a horizontal, vertical, or diagonal line is the winner.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

The main observation is 2 planes of the 3x3 board. For player_1, the first plane represents the placement of Xs, and the second plane shows the placement of Os. The possible values for each cell are 0 or 1; in the first plane, 1 indicates that an X has been placed in that cell, and 0 indicates
that X is not in that cell. Similarly, in the second plane, 1 indicates that an O has been placed in that cell, while 0 indicates that an O has not been placed. For player_2, the observation is the same, but Xs and Os swap positions, so Os are encoded in plane 1 and Xs in plane 2. This allows for
self-play.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

Each action from 0 to 8 represents placing either an X or O in the corresponding cell. The cells are indexed as follows:


 ```
0 | 3 | 6
_________

1 | 4 | 7
_________

2 | 5 | 8
 ```

### Rewards

| Winner | Loser |
| :----: | :---: |
| +1     | -1    |

If the game ends in a draw, both players will receive a reward of 0.

### Version History

* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""
