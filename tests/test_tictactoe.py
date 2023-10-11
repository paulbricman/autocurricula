from pelt.games.tictactoe import TicTacToe
from pelt.games.utils import action_ints_to_history
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytest


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_pretrained("distilgpt2")


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    return tok


def test_eval():
    game = TicTacToe()

    # Illegal by virtue of not being PZ-compatible
    history = action_ints_to_history([[0, 6, "illegal"]])
    assert game.eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, "illegal"]])
    assert game.eval(history) == [(0, -1)]

    # Illegal by game semantics
    history = action_ints_to_history([[0, 6, 500]])
    assert game.eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, 500]])
    assert game.eval(history) == [(0, -1)]

    # Partial legal games
    history = action_ints_to_history([[0, 6, 1]])
    assert game.eval(history) == [(0, 0)]

    history = action_ints_to_history([[0, 6, 1, 7]])
    assert game.eval(history) == [(0, 0)]

    # Full legal games
    history = action_ints_to_history([[0, 6, 1, 7, 2]])
    assert game.eval(history) == [(1, -1)]

    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    assert game.eval(history) == [(-1, 1)]

    # Multiple timelines with mixed outcomes
    history = action_ints_to_history([[0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]])
    assert game.eval(history) == [(1, -1), (-1, 1)]

    history = action_ints_to_history(
        [[0, 6, 1, 500], [0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]]
    )
    assert game.eval(history) == [(0, -1), (1, -1), (-1, 1)]


def test_preprocess():
    game = TicTacToe()

    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    contexts = game._preprocess(history)
    assert isinstance(contexts[0], str)


def test_act(model, tokenizer):
    game = TicTacToe()
    history = action_ints_to_history([[0, 6, 1, 7], [0, 6, 1, 7]])
    experiences = game.act(model, tokenizer, history)

    assert len(experiences) == 2  # two timelines
    assert len(experiences[0]) == 2  # one experience list, one action
    assert len(experiences[0][0]) == 2  # one thought experience, one action experience
    assert len(experiences[0][0][0]) == 2  # one context, one thought
