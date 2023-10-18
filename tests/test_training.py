from pelt.training import get_model_tok
from pelt.defaults import default_config
import pytest


@pytest.fixture
def config():
    return default_config()


@pytest.fixture
def model_tok(config):
    return get_model_tok("facebook/opt-125m", config)


def test_adapters(model_tok, config):
    peft_model, _ = model_tok
    peft_model.load_adapter(config["peft"]["path"], adapter_name="A1")
    peft_model.set_adapter(adapter_name="A1")

    assert list(peft_model.peft_config.keys()) == ["default", "A1"]
    assert peft_model.active_adapter == "A1"

    peft_model.load_adapter(config["peft"]["path"], adapter_name="A2")
    peft_model.set_adapter(adapter_name="A2")

    assert list(peft_model.peft_config.keys()) == ["default", "A1", "A2"]
    assert peft_model.active_adapter == "A2"

    peft_model.set_adapter(adapter_name="default")
    assert peft_model.active_adapter == "default"
