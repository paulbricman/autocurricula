from autocurricula.games.chess import preprocess, play
from autocurricula.games.pettingzoo_adapter import act, eval
from autocurricula.games.utils import action_ints_to_history
from autocurricula.self_play_trainer import SelfPlayTrainer
from autocurricula.self_play_config import SelfPlayConfig
from autocurricula.defaults import default_peft_config

from pettingzoo.classic import chess_v6

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
    history = action_ints_to_history([[3589, 4165, 77, 1245, 4173, 2413]])
    contexts = preprocess(history)
    assert isinstance(contexts[0], str)


def test_act(ac_trainer):
    history = action_ints_to_history([[3589, 4165, 77, 1245], [3589, 4165, 77, 1245]])
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
    chess_eval = lambda history: eval(history, chess_v6.env())

    # Illegal by virtue of not being PZ-compatible
    history = action_ints_to_history([["illegal"]])
    assert chess_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[3589, 4165, "illegal"]])
    assert chess_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[3589, 4165, 77, "illegal"]])
    assert chess_eval(history) == [(None, -math.inf)]

    # Illegal by game semantics
    history = action_ints_to_history([[3589, 4165, 999999]])
    assert chess_eval(history) == [(-math.inf, None)]

    history = action_ints_to_history([[3589, 4165, 77, 999999]])
    assert chess_eval(history) == [(None, -math.inf)]

    # Partial legal games
    history = action_ints_to_history([[3589, 4165, 77]])
    assert chess_eval(history) == [(0, 0)]

    history = action_ints_to_history([[3589, 4165, 77, 1245]])
    assert chess_eval(history) == [(0, 0)]

    # Multiple timelines with mixed outcomes
    history = action_ints_to_history([[3589, 4165, 999999], [3589, 4165, 77, 999999]])
    assert chess_eval(history) == [(-math.inf, None), (None, -math.inf)]

    # Full legal games
    history = action_ints_to_history(
        [
            [
                3005,
                2997,
                3563,
                2421,
                2343,
                4165,
                661,
                3070,
                3125,
                2954,
                2684,
                372,
                1837,
                3589,
                85,
                1253,
                1245,
                2343,
                1754,
                1022,
                1975,
                4238,
                1242,
                1767,
                2993,
                3071,
                663,
                669,
                1447,
                4311,
                1264,
                3563,
                2337,
                3581,
                3220,
                4105,
                1908,
                2998,
                4089,
                3128,
                1178,
                3725,
                4165,
                3683,
                883,
                1170,
                161,
                664,
                1475,
                1321,
                2421,
                2340,
                1754,
                745,
                12,
                1979,
                3581,
                4382,
                3516,
                3506,
                3653,
                2047,
                3589,
                1978,
                1244,
                4218,
                1886,
                672,
                157,
                645,
                4238,
                14,
                1172,
                1978,
                2930,
                2995,
                1242,
                2628,
                3742,
                1182,
                3529,
                2431,
                1905,
                2487,
                1318,
                2998,
                2578,
                3510,
                659,
                85,
                1198,
                2002,
                4528,
                1376,
                151,
                3129,
                664,
                4599,
                1391,
                4089,
                3505,
                2465,
                2924,
                3947,
                647,
                3511,
                1321,
                4363,
                2994,
                1829,
                2415,
                2380,
                1888,
                4161,
                1972,
                3129,
                14,
                1231,
                1174,
                2466,
                1788,
                4365,
                2118,
                4477,
                1571,
                2785,
                1394,
                3633,
                4478,
                1825,
                3123,
                1178,
                2126,
                2640,
                2997,
                2399,
                960,
                2775,
                2049,
                151,
                2030,
                2220,
                2557,
                3505,
                3083,
                4559,
                1318,
                2921,
                815,
                3216,
                1409,
                664,
                2048,
                1607,
                3148,
                1023,
                955,
                444,
                4311,
                1068,
                1977,
                2342,
                2572,
                3638,
                2121,
                226,
                1537,
                4575,
                223,
                4363,
                2190,
                468,
                1535,
                2921,
                1609,
                2793,
                1024,
                2556,
                1555,
                1985,
                514,
                3140,
                3299,
                2340,
                4457,
                2412,
                444,
                2558,
                1024,
                2491,
                4530,
                2340,
                4650,
                2415,
                4121,
                1315,
                1190,
                752,
                517,
                2518,
                1464,
                2993,
                2924,
                2890,
                2999,
                2423,
                296,
                3605,
                1101,
                3875,
                1680,
                4472,
                1101,
                4245,
                3610,
                2340,
                1679,
                4124,
                1025,
                2409,
                953,
                4472,
                687,
                4251,
                3018,
                2488,
                1025,
                3072,
                369,
                1753,
                703,
                3678,
                442,
            ]
        ]
    )
    assert chess_eval(history) == [(1, -1)]
