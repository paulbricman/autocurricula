from peft import LoraConfig, TaskType
from trl import PPOConfig


def default_peft_config():
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )


def default_ppo_config():
    return PPOConfig(ppo_epochs=1, batch_size=2, remove_unused_columns=False)
