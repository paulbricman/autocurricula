from peft import LoraConfig, TaskType, PeftConfig
from trl import PPOConfig


def default_peft_config() -> PeftConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
    )


def default_ppo_config() -> PPOConfig:
    return PPOConfig(ppo_epochs=1, batch_size=2, remove_unused_columns=False)
