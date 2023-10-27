from autocurricula import SelfPlayConfig, SelfPlayTrainer, set_player
from autocurricula.games.utils import is_integer

import math


def play(players, model, tokenizer):
    prompt = "Here's an integer: +"
    prompt_ids = tokenizer(prompt, return_tensors="pt")

    # Have each player issue a number.
    # Use `set_player` to activate the right player adapter.
    nums = []
    for player in players:
        set_player(model, player)
        # Ideally you'd run multiple games in parallel by batch generation.
        num_ids = model.generate(**prompt_ids, do_sample=True, max_new_tokens=1)
        num = tokenizer.decode(num_ids[0, len(prompt_ids[0]) :])
        nums += [num]

    # The rest of this function is just basic data manipulation.
    # Package up experiences for updating the players.
    experiences = [
        {
            "thoughts": [
                {
                    "context": prompt,
                    "behavior": num,
                }
            ],
            "action": num,
        }
        for num in nums
    ]

    if all([is_integer(num) for num in nums]):
        return [(int(nums[0]) - int(nums[1]), int(nums[1]) - int(nums[0]))], [
            experiences
        ]
    else:
        return [
            (
                None if is_integer(nums[0]) else -math.inf,
                None if is_integer(nums[1]) else -math.inf,
            )
        ], [experiences]
    return [], []


sp_config = SelfPlayConfig()
sp_trainer = SelfPlayTrainer(sp_config)
sp_trainer.train("facebook/opt-125m", play)

# Access the final list of players.
print(sp_trainer.players)

# Access the final ELO leaderboard.
print(sp_trainer.leaderboard)
