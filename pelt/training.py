from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from pelt.operators import league_entry, league_match
from pelt.defaults import default_config
from pelt.leaderboard import populate_leaderboard, update_leaderboard
from trl import PPOConfig
import json
import pprint


def update(model, league, matches, evals, history, config):
    sars = trajectories_by_model(league, matches, evals, history)

    config = PPOConfig()

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
            update(model, league, matches, evals, history, config)

    return league


def trajectories_by_model(league, matches, evals, histories):
    sars = {}
    for player in league:
        sars[json.dumps(player)] = []

    for match, eval_timelines, history_timelines in zip(matches, evals, histories):
        p1, p2 = match

        for eval, history in zip(eval_timelines, history_timelines):
            p1_states_actions = flatten(
                [e["thoughts"] for e_idx, e in enumerate(history) if e_idx % 2 == 0]
            )
            p1_states = [e["context"] for e in p1_states_actions]
            p1_actions = [e["behavior"] for e in p1_states_actions]

            p2_states_actions = flatten(
                [e["thoughts"] for e_idx, e in enumerate(history) if e_idx % 2 == 1]
            )
            p2_states = [e["context"] for e in p2_states_actions]
            p2_actions = [e["behavior"] for e in p2_states_actions]

            if len(p1_actions):
                p1_rewards = [eval[0] for _ in range(len(p1_actions))]
            else:
                p1_rewards = []

            if len(p2_actions):
                p2_rewards = [eval[1] for _ in range(len(p2_actions))]
            else:
                p2_rewards = []

            # https://huggingface.co/docs/trl/ppo_trainer#expected-dataset-format
            new_p1_experiences = [
                {"query": q, "completion": c, "reward": r}
                for q, c, r in zip(p1_states, p1_actions, p1_rewards)
            ]
            new_p2_experiences = [
                {"query": q, "completion": c, "reward": r}
                for q, c, r in zip(p2_states, p2_actions, p2_rewards)
            ]

            sars[json.dumps(p1)] += new_p1_experiences
            sars[json.dumps(p2)] += new_p2_experiences

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
    tokenizer.pad_token = tokenizer.eos_token

    return peft_model, tokenizer
