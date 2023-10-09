from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os


def train(model, act, eval, config):
    pass