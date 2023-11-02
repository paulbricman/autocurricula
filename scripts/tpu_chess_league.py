from autocurricula import LeagueTrainer
from autocurricula.games.chess import play

lt_trainer = LeagueTrainer()
lt_trainer.train("facebook/opt-125m", play)
