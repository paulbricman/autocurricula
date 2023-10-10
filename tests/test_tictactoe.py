from pelt.games.tictactoe import TicTacToe
from pelt.games.utils import action_ints_to_history


def test_eval():
    game = TicTacToe()

    history = action_ints_to_history([[0, 6, "illegal"]])
    assert game.eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, "illegal"]])
    assert game.eval(history) == [(0, -1)]

    history = action_ints_to_history([[0, 6, 500]])
    assert game.eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, 500]])
    assert game.eval(history) == [(0, -1)]

    history = action_ints_to_history([[0, 6, 1, 7, 2]])
    assert game.eval(history) == [(1, -1)]

    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    assert game.eval(history) == [(-1, 1)]
