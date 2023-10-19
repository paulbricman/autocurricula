from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from pelt.operators import league_entry, league_match
from pelt.defaults import default_config
from pelt.leaderboard import populate_leaderboard, update_leaderboard

import json


def update(league, matches, evals, history, config):
    return league


def train(
    model_name,
    play,
    match=league_match,
    entry=league_entry,
    update=update,
    config=default_config(),
):
    model, tokenizer = get_model_tok(model_name, config)
    league = []
    leaderboard = {}

    for generation in range(config["league"]["generations"]):
        entrants = entry(model, generation, config)
        league += entrants

        leaderboard = populate_leaderboard(leaderboard, entrants, config)
        populate_with_entrant_adapters(model, entrants, config)

        matches = match(league, generation, config)

        for _ in range(config["league"]["rounds"]):
            evals, history = zip(
                *[play(model, match, tokenizer, config) for match in matches]
            )

            leaderboard = update_leaderboard(leaderboard, matches, evals, config)
            league = update(league, matches, evals, history, config)

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
