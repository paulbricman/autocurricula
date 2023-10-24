import json


def new_elos(winner_elo, loser_elo, k=20):
    """
    Return new ELOs in light of a previous game, based on previous player ELOs.
    """
    winner_expected = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    loser_expected = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))

    winner_new_elo = winner_elo + k * (1 - winner_expected)
    loser_new_elo = loser_elo + k * (0 - loser_expected)

    return winner_new_elo, loser_new_elo


def populate_leaderboard(leaderboard, entrants, config):
    """
    Just add default ELO entries for entrants on the leaderboard.
    """
    for entrant in entrants:
        leaderboard[json.dumps(entrant)] = config["leaderboard"].get("default", 1000)
    return leaderboard


def update_leaderboard(leaderboard, matches, evals, config):
    """
    Given a host of matches and their outcomes, update the leaderboarb with new ELOs.
    """
    for match, timeline_evals in zip(matches, evals):
        for eval in timeline_evals:
            if eval[0] > eval[1]:
                winner, loser = match
            else:
                loser, winner = match

            leaderboard[json.dumps(winner)], leaderboard[json.dumps(loser)] = new_elos(
                leaderboard[json.dumps(winner)],
                leaderboard[json.dumps(loser)],
                k=config["leaderboard"].get("k", 20),
            )
    return leaderboard
