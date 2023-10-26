from autocurricula.defaults import default_peft_config, default_ppo_config
from autocurricula.autocurriculum_config import AutocurriculumConfig

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOTrainer, PPOConfig
from peft import PeftConfig
import torch

from typing import List, Tuple, Dict, Union, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
import json


disable_progress_bar()


class AutocurriculumTrainer(ABC):
    """
    Base class for `autocurricula` trainers. It handles the general logic of having model adaptations
    play against each other and learn from experience, delegating the `entry` and `match` logic to
    specialized trainers.
    """

    def __init__(self, ac_config: AutocurriculumConfig):
        assert isinstance(ac_config, AutocurriculumConfig)
        self.ac_config = ac_config
        self.current_gen = 0
        self.players = []
        self.leaderboard = {}

    @abstractmethod
    def entry(self):
        """
        In the current generation, which new players join?
        Returns list of individual player dicts acting as "specs."
        """
        pass

    @abstractmethod
    def match(self):
        """
        In the current generation, and given the current league who plays who?
        Returns list of matchups (i.e. tuples of player dicts).
        `play` will subsequently be called using these.
        """
        pass

    def train(
        self,
        model: Union[str, AutoModelForCausalLM],
        play: Callable,
        peft_config: PeftConfig = default_peft_config(),
        ppo_config: PPOConfig = default_ppo_config(),
    ):
        """
        High-level method orchestrating the whole process of having models play each other
        and updating their respective adapters in light of their experiences.
        """
        self.pin_model_and_tok(model, peft_config)
        self.peft_config, self.ppo_config = peft_config, ppo_config

        for self.current_gen in tqdm(range(self.ac_config.generations), "generation"):
            self.accommodate_entrants(self.entry())
            matches = self.match()

            for _ in tqdm(range(self.ac_config.rounds), "round"):
                evals_histories = [play(m, self.model, self.tokenizer) for m in matches]
                evals, histories = zip(*evals_histories)

                self.update_players(matches, evals, histories)
                self.update_leaderboard(matches, evals)

    def accommodate_entrants(self, entrants: List[Dict]):
        # TODO: Force a "gen" field on the entrants so that users don't have to.
        self.players += entrants

        for entrant in entrants:
            self.leaderboard[json.dumps(entrant)] = 1000

        for entrant in entrants:
            self.model.pretrained_model.load_adapter(
                "adapter_params", adapter_name=json.dumps(entrant)
            )

    def update_leaderboard(self, matches: List[Tuple], evals: List[List[Tuple]]):
        """
        Given a host of matches and their outcomes, update the leaderboarb with new ELOs.
        """
        assert all(
            [len(m) == 2 for m in matches]
        ), "Multiplayer games aren't currently supported. But this is where they'd be!"

        def new_elos(winner_elo, loser_elo):
            winner_expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
            loser_expected = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
            winner_new_elo = winner_elo + 20 * (1 - winner_expected)
            loser_new_elo = loser_elo + 20 * (0 - loser_expected)
            return winner_new_elo, loser_new_elo

        for match, match_evals in zip(matches, evals):
            # There are multiple games played in parallel for throughput.
            # Each match yields a whole batch of evals.
            for eval in match_evals:
                if (
                    eval[0] != eval[1]
                    and all([e is not None for e in eval])
                    and match[0] != match[1]
                ):
                    if eval[0] > eval[1]:
                        winner, loser = match
                    else:
                        loser, winner = match

                    (
                        self.leaderboard[json.dumps(winner)],
                        self.leaderboard[json.dumps(loser)],
                    ) = new_elos(
                        self.leaderboard[json.dumps(winner)],
                        self.leaderboard[json.dumps(loser)],
                    )

    def update_players(
        self, matches: List[Tuple], evals: List[List[Tuple]], history: List[List[Dict]]
    ):
        """
        Given self-contained objects containing info on who played who, how they played,
        and with what outcomes, manage the whole process of updating players in light of
        their experiences.
        """
        sars = self.trajectories_by_player(matches, evals, history)

        for player in self.players:
            if len(sars[json.dumps(player)]) < 1 or player["gen"] != self.current_gen:
                continue

            dataset = Dataset.from_list(sars[json.dumps(player)])
            dataset = self.pretok(dataset, self.tokenizer)
            self.model.pretrained_model.set_adapter(adapter_name=json.dumps(player))

            ppo_trainer = PPOTrainer(
                model=self.model,
                config=self.ppo_config,
                dataset=dataset,
                tokenizer=self.tokenizer,
            )

            for batch in ppo_trainer.dataloader:
                if not batch:
                    continue

                # For some reason ppo_trainer transposes tokens, so transpose them back.
                batch["query_ids"] = list(torch.stack(batch["query_ids"]).T)
                batch["completion_ids"] = list(torch.stack(batch["completion_ids"]).T)
                batch["reward"] = list(batch["reward"].to(torch.bfloat16))

                ppo_trainer.step(
                    batch["query_ids"],
                    batch["completion_ids"],
                    batch["reward"],
                )

    def trajectories_by_player(
        self,
        matches: List[Tuple],
        evals: List[List[Tuple]],
        histories: List[List[Dict]],
    ):
        """
        Merge self-contained objects containing info on who played who, how they played,
        and with what outcomes into a unified dict where keys are players and values are
        lists of state-action-reward dicts on which to later train on.
        """

        def flatten(l):
            return [item for sublist in l for item in sublist]

        sars = {}
        for player in self.players:
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

                new_p1_experiences = [
                    e for e in new_p1_experiences if e["reward"] is not None
                ]
                new_p2_experiences = [
                    e for e in new_p2_experiences if e["reward"] is not None
                ]

                sars[json.dumps(p1)] += new_p1_experiences
                sars[json.dumps(p2)] += new_p2_experiences

        return sars

    def pretok(self, dataset: Dataset, tokenizer: AutoTokenizer):
        """
        Tokenize the texts which comprise the dataset on which the players are trained on in advance.
        """

        def tokenize(sample):
            sample["query_ids"] = tokenizer.encode(
                sample["query"], truncation=True, padding="max_length", max_length=16
            )
            sample["completion_ids"] = tokenizer.encode(
                sample["completion"],
                truncation=True,
                padding="max_length",
                max_length=16,
            )
            return sample

        return dataset.map(tokenize, batched=False)

    def pin_model_and_tok(self, model: AutoModelForCausalLM, peft_config: PeftConfig):
        """
        Load `trl`-wrapped `peft`-wrapped model and tokenizer based on `transformers` model (name).
        """
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model, peft_config=peft_config
        )
        self.model.pretrained_model.save_pretrained("adapter_params")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.pretrained_model.config._name_or_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
