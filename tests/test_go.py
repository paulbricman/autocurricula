from autocurricula.games.go import preprocess, play
from autocurricula.games.pettingzoo_adapter import act, eval
from autocurricula.games.utils import action_ints_to_history
from autocurricula.self_play_trainer import SelfPlayTrainer
from autocurricula.self_play_config import SelfPlayConfig
from autocurricula.defaults import default_peft_config

from pettingzoo.classic import go_v5

import pytest
import math


@pytest.fixture
def ac_trainer():
    ac_config = SelfPlayConfig(
        epochs=2,
        rounds=2,
        matches=2,
    )
    ac_trainer = SelfPlayTrainer(ac_config)
    ac_trainer.pin_model_and_tok("facebook/opt-125m", default_peft_config())
    return ac_trainer


def test_preprocess():
    history = action_ints_to_history([[15, 21, 16, 10, 0], [15, 21, 16, 10]])
    contexts = preprocess(history)
    assert isinstance(contexts[0], str)


def test_act(ac_trainer):
    history = action_ints_to_history([[15, 21, 16, 10, 0], [15, 21, 16, 10]])
    timelines = act(ac_trainer.model, ac_trainer.tokenizer, history, preprocess)

    assert len(timelines) == 2
    assert len(timelines[0]["thoughts"]) == 2


def test_play(ac_trainer):
    # Pretend a toy model is actually two models playing.
    evals, _ = play(
        ["default", "default"],
        ac_trainer.model,
        ac_trainer.tokenizer,
    )
    assert all([e == (-math.inf, None) for e in evals])


def test_train(ac_trainer):
    ac_trainer.train("facebook/opt-125m", play)


def test_eval():
    go_eval = lambda history: eval(history, go_v5.env(board_size=9))

    # Illegal by virtue of not being PZ-compatible
    history = action_ints_to_history([["illegal"]])
    assert go_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[15, 21, "illegal"]])
    assert go_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[15, 21, 16, "illegal"]])
    assert go_eval(history) == [(None, -math.inf)]

    # Illegal by game semantics
    history = action_ints_to_history([[15, 21, 500]])
    assert go_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[15, 21, 16, 500]])
    assert go_eval(history) == [(None, -math.inf)]

    # Partial legal games
    history = action_ints_to_history([[15, 21]])
    assert go_eval(history) == [(0, 0)]

    history = action_ints_to_history([[15, 21, 16]])
    assert go_eval(history) == [(0, 0)]

    # Multiple timelines with mixed outcomes
    history = action_ints_to_history([[15, 21, 500], [15, 21, 16]])
    assert go_eval(history) == [(-math.inf, None), (0, 0)]

    # Full legal games
    history = action_ints_to_history(
        [
            [
                67,
                22,
                31,
                39,
                69,
                7,
                12,
                11,
                68,
                49,
                60,
                59,
                1,
                3,
                66,
                27,
                55,
                57,
                47,
                38,
                23,
                73,
                56,
                13,
                76,
                44,
                48,
                65,
                28,
                8,
                16,
                4,
                43,
                61,
                81,
                74,
                5,
                9,
                2,
                75,
                53,
                17,
                72,
                21,
                19,
                20,
                46,
                63,
                0,
                35,
                77,
                41,
                81,
                81,
            ]
        ]
    )
    assert go_eval(history) == [(-1, 1)]

    history = action_ints_to_history(
        [
            [
                59,
                56,
                54,
                39,
                13,
                26,
                80,
                17,
                28,
                32,
                71,
                22,
                45,
                23,
                46,
                60,
                16,
                48,
                52,
                53,
                35,
                78,
                58,
                15,
                55,
                8,
                68,
                18,
                47,
                4,
                40,
                51,
                77,
                61,
                21,
                10,
                81,
                30,
                0,
                9,
                31,
                67,
                73,
                7,
                43,
                1,
                79,
                3,
                11,
                38,
                76,
                24,
                42,
                75,
                49,
                29,
                25,
                65,
                14,
                37,
                74,
                50,
                2,
                41,
                5,
                6,
                36,
                64,
                19,
                72,
                34,
                63,
                12,
                57,
                70,
                20,
                33,
                44,
                34,
                3,
                33,
                66,
                52,
                16,
                4,
                35,
                81,
                42,
                74,
                27,
                25,
                55,
                81,
                45,
                69,
                73,
                46,
                28,
                62,
                36,
                43,
                61,
                22,
                53,
                60,
                23,
                44,
                6,
                24,
                74,
                81,
                8,
                41,
                42,
                26,
                17,
                3,
                54,
                16,
                19,
                35,
                50,
                15,
                81,
                53,
                0,
                7,
                81,
                17,
                81,
                78,
                47,
                46,
                30,
                75,
                63,
                61,
                18,
                72,
                38,
                20,
                67,
                74,
                73,
                6,
                66,
                56,
                54,
                10,
                37,
                36,
                19,
                28,
                45,
                64,
                65,
                9,
                1,
                0,
                72,
                1,
                81,
                55,
                54,
                63,
                39,
                81,
                57,
                73,
                81,
                29,
                47,
                51,
                81,
                8,
                81,
                42,
                81,
                50,
                81,
                32,
                81,
                23,
                81,
                81,
            ]
        ]
    )
    assert go_eval(history) == [(1, -1)]
