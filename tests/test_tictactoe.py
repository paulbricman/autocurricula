from autocurricula.games.tictactoe import preprocess
from autocurricula.games.utils import action_ints_to_history
from autocurricula.games.pettingzoo_adapter import play, act, eval
from autocurricula.league_trainer import LeagueTrainer
from autocurricula.league_config import LeagueConfig
from autocurricula.defaults import default_peft_config

from pettingzoo.classic import tictactoe_v3

import pytest
import math


@pytest.fixture
def ac_trainer():
    ac_config = LeagueConfig(
        epochs=4,
        rounds=2,
        matches=10,
        ma_weight=0.4,
        me_weight=0.2,
        le_weight=0.4,
    )
    ac_trainer = LeagueTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_eval():
    tictactoe_eval = lambda history: eval(history, tictactoe_v3.env())

    # Illegal by virtue of not being PZ-compatible
    history = action_ints_to_history([["illegal"]])
    assert tictactoe_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[0, 6, "illegal"]])
    assert tictactoe_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[0, 6, 1, "illegal"]])
    assert tictactoe_eval(history) == [(None, -math.inf)]

    # Illegal by game semantics
    history = action_ints_to_history([[0, 6, 500]])
    assert tictactoe_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[0, 6, 1, 500]])
    assert tictactoe_eval(history) == [(None, -math.inf)]

    # Partial legal games
    history = action_ints_to_history([[0, 6, 1]])
    assert tictactoe_eval(history) == [(0, 0)]

    history = action_ints_to_history([[0, 6, 1, 7]])
    assert tictactoe_eval(history) == [(0, 0)]

    # Full legal games
    history = action_ints_to_history([[0, 6, 1, 7, 2]])
    assert tictactoe_eval(history) == [(1, -1)]

    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    assert tictactoe_eval(history) == [(-1, 1)]

    # Multiple timelines with mixed outcomes
    history = action_ints_to_history([[0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]])
    assert tictactoe_eval(history) == [(1, -1), (-1, 1)]

    history = action_ints_to_history(
        [[0, 6, 1, 500], [0, 6, 1, 7, 2], [0, 6, 1, 7, 3, 8]]
    )
    assert tictactoe_eval(history) == [(None, -math.inf), (1, -1), (-1, 1)]


def test_preprocess():
    history = action_ints_to_history([[0, 6, 1, 7, 3, 8]])
    contexts = preprocess(history)
    assert isinstance(contexts[0], str)


def test_act(ac_trainer):
    history = action_ints_to_history([[0, 6, 1, 7], [0, 6, 1, 7]])
    timelines = act(ac_trainer.model, ac_trainer.tokenizer, history, preprocess)

    assert len(timelines) == 2
    assert len(timelines[0]["thoughts"]) == 2


def test_play(ac_trainer):
    # Pretend a toy model is actually two models playing.
    evals, _ = play(
        ["default", "default"],
        ac_trainer.model,
        ac_trainer.tokenizer,
        preprocess,
        tictactoe_v3.env(),
    )
    assert all([e == (-math.inf, None) for e in evals])
