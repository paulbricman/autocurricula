from autocurricula.games.utils import set_player

from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from typing import List, Dict, Tuple


def play(
    match: Tuple[Dict],
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
) -> Tuple[List[Tuple], List[List[Dict]]]:
    batch_size = 2
    num_batches = 1

    dataset = (
        load_dataset("hendrycks/competition_math")["train"]
        .to_iterable_dataset()
        .shuffle(seed=0)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    for e_idx, e in enumerate(dataloader):
        if e_idx > num_batches:
            break

        # Generate d_thought.
        prompts = d_format(e["problem"], e["solution"])
        prompts = [
            f"{f}...\n\nIV. Here is a final verdict on whether the solution is appropriate (YES/NO):"
            for f in prompts
        ]
        # Generate d_verdict.
        # Compose d_gt experiences.

        # Generate g solution.
        # Generate d_thought.
        g_solution = None
        prompts = d_format(e["problem"], g_solution)
        prompts = [
            f"{f}...\n\nIV. Here is a final verdict on whether the solution is appropriate (YES/NO):"
            for f in prompts
        ]
        # Generate d_verdict.
        # Compose d_g experiences.
        # Compose g experiences.

    # Return evals, history


def d_format(problems, solutions):
    prompts = []
    for problem, solution in zip(problems, solutions):
        prompts += [
            f"""
I. Here is a math problem:

{problem}

II. Here is a candidate solution to the above math problem:

{solution}

III. Here is a general evaluation of correctness and general suitability of the candidate solution in the context of the original math problem:

"""
        ]

    return prompts
