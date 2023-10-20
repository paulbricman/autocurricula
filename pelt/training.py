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


def trajectories_by_model(league, matches, evals, history):
    sars = {}
    for player in league:
        sars[json.dumps(player)] = []

    for match, eval_timelines, history_timelines in zip(matches, evals, history):
        p1, p2 = match

        p1_states_actions = [
            e["thoughts"] for e_idx, e in enumerate(history_timelines) if e_idx % 2 == 0
        ]
        p1_states = [[e["context"] for e in f] for f in p1_states_actions]
        p1_actions = [[e["behavior"] for e in f] for f in p1_states_actions]

        p2_states_actions = [
            e["thoughts"] for e_idx, e in enumerate(history_timelines) if e_idx % 2 == 1
        ]
        p2_states = [[e["context"] for e in f] for f in p2_states_actions]
        p2_actions = [[e["behavior"] for e in f] for f in p2_states_actions]

        if len(p1_actions):
            p1_rewards = [
                [e[0] for _ in range(len(p1_actions[0]))] for e in eval_timelines
            ]
        else:
            p1_rewards = []

        if len(p2_actions):
            p2_rewards = [
                [e[1] for _ in range(len(p2_states[0]))] for e in eval_timelines
            ]
        else:
            p2_rewards = []

        p1_states = flatten(p1_states)
        p1_actions = flatten(p1_actions)
        p1_rewards = flatten(p1_rewards)
        p2_states = flatten(p2_states)
        p2_actions = flatten(p2_actions)
        p2_rewards = flatten(p2_rewards)

        sars[json.dumps(p1)] += list(zip(p1_states, p1_actions, p1_rewards))
        sars[json.dumps(p2)] += list(zip(p2_states, p2_actions, p2_rewards))

    return sars


def flatten(l):
    return [item for sublist in l for item in sublist]


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
