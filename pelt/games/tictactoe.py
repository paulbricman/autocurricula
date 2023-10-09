from game import Game
from abc import ABC, abstractmethod
from typing import List, Tuple


class TicTacToe(Game):
    def act(self, history: List[List[Tuple[str, str, str]]]):
        # Implement tictactoe-specific act logic
        pass

    def eval(self, history: List[List[Tuple[str, str, str]]]):
        # Implement tictactoe-specific eval logic
        pass
