from autocurricula import SelfPlayTrainer, SelfPlayConfig
from autocurricula.games.chess import play

lt_config = SelfPlayConfig(epochs=1, rounds=1, matches=1)
lt_trainer = SelfPlayTrainer()
lt_trainer.train("facebook/opt-125m", play)
