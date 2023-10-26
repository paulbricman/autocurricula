from autocurricula import SelfPlayConfig, SelfPlayTrainer
from autocurricula.games.utils import set_player


def play(players, model, tokenizer):
    prompt = " "
    prompt_ids = tokenizer(prompt, return_tensors="pt")

    # Have each player produce a string.
    strs = []
    for player in players:
        # Use `set_player` to activate the right player adapter.
        set_player(model, player)
        # Ideally you'd run multiple games in parallel by batch generation.
        str_ids = model.generate(**prompt_ids, do_sample=True, max_new_tokens=3)[0]
        strs += [tokenizer.decode(str_ids)]

    # The rest of this function is just basic data manipulation.
    # Package up experiences for updating the players.
    experiences = [
        {
            "thoughts": [
                {
                    "context": prompt,
                    "behavior": str,
                }
            ],
            "action": str,
        }
        for str in strs
    ]

    return [(strs[0] > strs[1], strs[0] < strs[1])], [experiences]


sp_config = SelfPlayConfig()
sp_trainer = SelfPlayTrainer(sp_config)
sp_trainer.train("facebook/opt-125m", play)

# Access the final list of players.
print(sp_trainer.players)

# Access the final ELO leaderboard.
print(sp_trainer.leaderboard)
