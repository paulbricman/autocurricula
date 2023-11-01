from autocurricula.games.go import preprocess
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
