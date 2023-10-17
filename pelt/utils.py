from transformers import AutoModel, AutoTokenizer


def to_actual_model_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
