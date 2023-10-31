from trl import AutoModelForCausalLMWithValueHead

from typing import List, Dict, Tuple
import math
import json


def set_player(model: AutoModelForCausalLMWithValueHead, player: Dict):
    if isinstance(player, dict):
        player = json.dumps(player)

    model.pretrained_model.set_adapter(player)


def action_ints_to_history(history: List[List[Dict]]):
    return [[{"action": action} for action in timeline] for timeline in history]
