def default_config():
    return {
        "league": {
            "n_generations": 4,
            "n_matches": 16,
            "n_rounds": 4,
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
            "task_type": "CAUSAL_LM",
            "kwargs": {},
        },
    }
