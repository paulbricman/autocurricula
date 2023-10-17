from pelt.utils import to_actual_model_tokenizer
from pelt.training import update
from pelt.defaults import league_entry, league_match, default_config


def league(
    model,
    play,
    match=league_match,
    entry=league_entry,
    config=default_config(),
    tokenizer=None,
):
    if isinstance(model, str):
        model, tokenizer = to_actual_model_tokenizer(model)

    league = []
    for generation in range(config["league"]["n_generations"]):
        league += entry(generation, config)
        matches = match(league, generation, config)

        for round in range(config["league"]["n_rounds"]):
            evals_and_history = [play(match, tokenizer, config) for match in matches]
            league = update(league, matches, evals_and_history, config)

    return league
