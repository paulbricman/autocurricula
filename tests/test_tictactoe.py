from pelt.games.tictactoe import play, act, eval, preprocess
from pelt.games.utils import action_ints_to_history
from pelt.defaults import default_config
from transformers import AutoTokenizer, AutoModelForCausalLM
import pytest
from ml_collections.config_dict import ConfigDict
import pprint


pp = pprint.PrettyPrinter()


@pytest.fixture
def model():
    return AutoModelForCausalLM.from_pretrained("distilgpt2")


@pytest.fixture
def tokenizer():
    tok = AutoTokenizer.from_pretrained("distilgpt2")
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def config():
    return default_config()


def test_eval():
    # Illegal by virtue of not being PZ-compatible
    history = action_ints_to_history([["illegal"]])
    assert eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, "illegal"]])
    assert eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, "illegal"]])
    assert eval(history) == [(0, -1)]

    # Illegal by game semantics
    history = action_ints_to_history([[0, 6, 500]])
    assert eval(history) == [(-1, 0)]

    history = action_ints_to_history([[0, 6, 1, 500]])
    assert eval(history) == [(0, -1)]

    # Partial legal games
    history = action_ints_to_history([[0, 6, 1]])
    assert eval(history) == [(0, 0)]

    history = action_ints_to_history([[0, 6, 1, 7]])
    assert eval(history) == [(0, 0)]

    # Full legal games
    history = action_ints_to_history([[0, 6, 1, 7, 2]])
    assert eval(history) == [(1, -1)]

    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    assert eval(history) == [(-1, 1)]

    # Multiple timelines with mixed outcomes
    history = action_ints_to_history([[0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]])
    assert eval(history) == [(1, -1), (-1, 1)]

    history = action_ints_to_history(
        [[0, 6, 1, 500], [0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]]
    )
    assert eval(history) == [(0, -1), (1, -1), (-1, 1)]


def test_preprocess():
    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    contexts = preprocess(history)
    assert isinstance(contexts[0], str)


def test_act(model, tokenizer):
    history = action_ints_to_history([[0, 6, 1, 7], [0, 6, 1, 7]])
    timelines = act(model, tokenizer, history)

    assert len(timelines) == 2  # two timelines
    assert len(timelines[0]["thoughts"]) == 2  # two thoughts behind an action


def test_play(model, tokenizer, config):
    # Pretend a toy model is actually two models playing.
    evals, history = play([model, model], tokenizer, config)
    assert evals == [(-1, 0), (-1, 0)]
