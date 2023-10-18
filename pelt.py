from pelt.training import (
    update,
    populate_with_entrant_adapters,
    peft_model_tok,
)
from pelt.operators import league_entry, league_match
from pelt.defaults import default_config


def league(
    model_name, play, match=league_match, entry=league_entry, config=default_config()
):
    model, tokenizer = peft_model_tok(model_name, config)

    league = []
    for generation in range(config["league"]["n_generations"]):
        entrants = entry(model, generation, config)
        populate_with_entrant_adapters(model, entrants, config)
        matches = match(league, generation, config)

        for round in range(config["league"]["n_rounds"]):
            evals_and_history = [
                play(model, match, tokenizer, config) for match in matches
            ]
            league = update(league, matches, evals_and_history, config)

    return league
