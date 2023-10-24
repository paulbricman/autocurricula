from peft import TaskType, get_peft_model, LoraConfig, PeftModel
from pelt.operators import league_entry, league_match
from pelt.defaults import default_config
from pelt.leaderboard import populate_leaderboard, update_leaderboard

from trl import PPOConfig, PPOTrainer
from datasets import Dataset
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

import json


def pretok(dataset, tokenizer):
    """
    Tokenize the texts which comprise the dataset on which the players are trained on in advance.
    """

    def tokenize(sample):
        sample["query_ids"] = tokenizer.encode(
            sample["query"], truncation=True, padding="max_length", max_length=16
        )
        sample["completion_ids"] = tokenizer.encode(
            sample["completion"], truncation=True, padding="max_length", max_length=16
        )
        return sample

    return dataset.map(tokenize, batched=False)


def trajectories_by_model(league, matches, evals, histories):
    """
    Merge self-contained objects containing info on who played who, how they played,
    and with what outcomes into a unified dict where keys are players and values are
    lists of state-action-reward dicts on which to later train on.
    """

    def flatten(l):
        return [item for sublist in l for item in sublist]

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


def update(model, tokenizer, league, matches, evals, history, config):
    """
    Given self-contained objects containing info on who played who, how they played,
    and with what outcomes, manage the whole process of updating players in light of
    their experiences.
    """
    sars = trajectories_by_model(league, matches, evals, history)

    for player in league:
        if len(sars[json.dumps(player)]) < 1:
            continue

        dataset = Dataset.from_list(sars[json.dumps(player)])
        dataset = pretok(dataset, tokenizer)
        ppo_config = PPOConfig(
            ppo_epochs=1, batch_size=min(2, len(dataset)), remove_unused_columns=False
        )
        model.pretrained_model.set_adapter(adapter_name=json.dumps(player))

        ppo_trainer = PPOTrainer(
            model=model,
            config=ppo_config,
            dataset=dataset,
            tokenizer=tokenizer,
        )

        for batch in ppo_trainer.dataloader:
            # For some reason ppo_trainer transposes tokens.
            batch["query_ids"] = list(torch.stack(batch["query_ids"]).T)
            batch["completion_ids"] = list(torch.stack(batch["completion_ids"]).T)
            batch["reward"] = list(batch["reward"].to(torch.bfloat16))

            stats = ppo_trainer.step(
                batch["query_ids"],
                batch["completion_ids"],
                batch["reward"],
            )


def populate_with_entrant_adapters(model, entrants, config):
    """
    Actually go and create new adapters in the backbone model for new players.
    """
    for entrant in entrants:
        model.pretrained_model.load_adapter(
            config["peft"]["path"], adapter_name=json.dumps(entrant)
        )


def train(
    model_name,
    play,
    match=league_match,
    entry=league_entry,
    update=update,
    config=default_config(),
):
    """
    High-level method orchestrating the whole process of having models play each other
    and updating their respective adapters in light of their experiences.
    """

    model, tokenizer = get_model_tok(model_name, config)
    league = []
    leaderboard = {}

    for generation in range(config["league"]["generations"]):
        entrants = entry(generation, config)
        league += entrants

        leaderboard = populate_leaderboard(leaderboard, entrants, config)
        populate_with_entrant_adapters(model, entrants, config)

        matches = match(league, generation, config)

        for _ in range(config["league"]["rounds"]):
            evals, history = zip(
                *[play(model, match, tokenizer, config) for match in matches]
            )

            leaderboard = update_leaderboard(leaderboard, matches, evals, config)
            update(model, tokenizer, league, matches, evals, history, config)

    return league


def get_peft_config(config):
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
    peft_config = get_peft_config(config)
    trl_peft_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name, peft_config=peft_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return trl_peft_model, tokenizer
