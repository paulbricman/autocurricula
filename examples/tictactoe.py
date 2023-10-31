from autocurricula import SelfPlayTrainer
from autocurricula.games.tictactoe import play

sp_trainer = SelfPlayTrainer()
sp_trainer.train("facebook/opt-125m", play)
