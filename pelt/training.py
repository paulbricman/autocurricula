from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
import json


def update(league, matches, evals_and_history, config):
    return league


def populate_with_entrant_adapters(model, entrants, config):
    for entrant in entrants:
        model.load_adapter(config["peft"]["path"], adapter_name=json.dumps(entrant))


def get_peft_config(config):
    # You'd think `PeftConfig` could handle all supported peft methods.
    # However, `peft` codebase itself still has rough edges.
    if config["peft"]["peft_type"] == "LORA":
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **config["peft"]["kwargs"],
        )

    assert False, f'{config["peft"]["peft_type"]} not available.'


def get_model_tok(model_name, config):
    """
    Given model name and `pelt` config, return `peft`-wrapped backbone model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = get_peft_config(config)
    peft_model = get_peft_model(model, peft_config)
    peft_model.save_pretrained(config["peft"]["path"])
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_model = PeftModel.from_pretrained(model, config["peft"]["path"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return peft_model, tokenizer
