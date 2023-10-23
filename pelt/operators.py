import random


def league_entry(generation, config):
    """
    Default league training entry logic by generation.
    """
    entrants = [
        {
            "role": "main_agent",
            "generation": generation,
        }
    ]
    if generation >= 1:
        entrants += [
            {
                "role": "main_exploiter",
                "generation": generation,
            }
        ]
    if generation >= 2:
        entrants += [
            {
                "role": "league_exploiter",
                "generation": generation,
            }
        ]
    return entrants


def league_match(league, generation, config):
    """
    Default league training matchmaking logic by generation.
    """
    main_agents = [p for p in league if p["role"] == "main_agent"]
    latest_main_agent = [p for p in main_agents if p["generation"] == generation][0]

    # Latest main agent plays themselves and all past versions.
    ma_ma_matches = [(latest_main_agent, main_agent) for main_agent in main_agents]

    legal_match_types = [ma_ma_matches]

    if generation >= 1:
        main_exploiters = [p for p in league if p["role"] == "main_exploiter"]
        latest_main_exploiter = [
            p for p in main_exploiters if p["generation"] == generation
        ][0]

        # Latest main exploiter plays latest main agent.
        me_ma_matches = [(latest_main_exploiter, latest_main_agent)]

        # Latest main agent plays latest main exploiter.
        ma_me_matches = [(latest_main_agent, latest_main_exploiter)]

        legal_match_types += [me_ma_matches, ma_me_matches]

    if generation >= 2:
        league_exploiters = [p for p in league if p["role"] == "league_exploiter"]
        latest_league_exploiter = [
            p for p in league_exploiters if p["generation"] == generation
        ]

        # Latest league exploiter plays the entire league except itself.
        le_matches = [
            (latest_league_exploiter, p)
            for p in league
            if p is not latest_league_exploiter
        ]

        # Latest main agent plays latest league exploiter.
        ma_le_matches = [(latest_main_agent, latest_main_exploiter)]

        legal_match_types += [le_matches, ma_le_matches]

    matches = []
    for match_id in range(config["league"]["matches"]):
        type = random.choices(
            range(len(legal_match_types)),
            weights=[
                config["league"]["weights"]["ma_ma"],
                config["league"]["weights"]["me_ma"],
                config["league"]["weights"]["ma_me"],
                config["league"]["weights"]["le"],
                config["league"]["weights"]["ma_le"],
            ][: len(legal_match_types)],
        )[0]

        matches += random.choices(legal_match_types[type])

    return matches
