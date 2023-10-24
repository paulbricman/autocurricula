def default_config():
    return {
        "leaderboard": {"default": 1000},
        "league": {
            "generations": 2,
            "matches": 4,
            "rounds": 2,
            "weights": {
                "me_ma": 0.2,
                "le": 0.2,
                "ma_ma": 0.2,
                "ma_me": 0.2,
                "ma_le": 0.2,
            },
        },
        "game": {"batch_size": 2},
        "peft": {
            "path": "./data",
            "peft_type": "LORA",
            "kwargs": {},
        },
    }
