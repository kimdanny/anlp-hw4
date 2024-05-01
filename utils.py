# # https://python.langchain.com/docs/integrations/text_embedding/huggingfacehub
# from langchain_community.embeddings import (
#     HuggingFaceEmbeddings,
#     HuggingFaceInstructEmbeddings,
# )
# from typing import Union


# def load_hf_embeddings(
#     model_path: str, instruct_model=False, device: str = "cpu"
# ) -> Union[HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings]:
#     model_kwargs = {"device": device}

#     # Initialize an Embedding instance with the specified parameters
#     if instruct_model:
#         embeddings = HuggingFaceInstructEmbeddings(
#             model_name=model_path,  # Provide the pre-trained model's path
#             model_kwargs=model_kwargs,  # Pass the model configuration options
#             encode_kwargs={"normalize_embeddings": True},  # Pass the encoding options
#         )
#     else:
#         embeddings = HuggingFaceEmbeddings(
#             model_name=model_path,  # Provide the pre-trained model's path
#             model_kwargs=model_kwargs,  # Pass the model configuration options
#             encode_kwargs={"normalize_embeddings": False},  # Pass the encoding options
#         )
#     return embeddings


models_info = {
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "hf_pipeline_task": "text-generation",
    },
    "llama2_7b_chat": {
        "model_id": "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf",
        "hf_pipeline_task": "text-generation",
    },
    "llama2_70b": {
        "model_id": "meta-llama/Llama-2-70b-hf",
        "hf_pipeline_task": "text-generation",
    },
    "flanT5Base": {
        "model_id": "google/flan-t5-base",
        "hf_pipeline_task": "text2text-generation",
    },
    "flanT5XXL": {
        "model_id": "google/flan-t5-xxl",
        "hf_pipeline_task": "text2text-generation",
    },
    "flanUl2": {
        "model_id": "google/flan-ul2",
        "hf_pipeline_task": "text2text-generation",
    },
}


def trim_sentence_by_token(sentence: str, tokenizer, use_model_max_length=False) -> str:
    # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize the sentence to check its length
    tokens = tokenizer.tokenize(sentence)
    if use_model_max_length:
        max_len = tokenizer.model_max_length
    else:
        max_len = 256

    # If the sentence exceeds the maximum length, truncate the tokens
    if len(tokens) > max_len:
        # Truncate the tokens to the max length
        truncated_tokens = tokens[:max_len]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    else:
        return sentence


def trim_sentence_by_token_len(sentence: str, tokenizer, max_tok_len) -> str:
    """
    Take sentence and tokenize using the tokenizer
    and returns the truncated text if the token length of the sentence exceeds max_tok_len
    """
    tokens = tokenizer.tokenize(sentence)

    # If the sentence exceeds the maximum length, truncate the tokens
    if len(tokens) > max_tok_len:
        # Truncate the tokens to the max length
        truncated_tokens = tokens[:max_tok_len]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text
    else:
        return sentence


def get_tokenized_length(sentence: str, tokenizer) -> int:
    tokens = tokenizer.tokenize(sentence)
    return len(tokens)
