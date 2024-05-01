# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/LaMP/rank_profiles.py

import torch
from transformers import AutoModel, AutoTokenizer
import json
from tqdm import tqdm
import argparse
import os

# Adapted bm25
from bm25 import BM25Okapi

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def extract_strings_between_quotes(input_string):
    output_list = []
    inside_quotes = False
    current_string = ""

    for char in input_string:
        if char == '"' and not inside_quotes:
            inside_quotes = True
        elif char == '"' and inside_quotes:
            inside_quotes = False
            output_list.append(current_string)
            current_string = ""
        elif inside_quotes:
            current_string += char

    return output_list


def extract_after_article(input_string):
    article_index = input_string.find("article:")
    if article_index == -1:
        return None
    return input_string[article_index + len("article:") :].strip()


def extract_after_description(input_string):
    article_index = input_string.find("description:")
    if article_index == -1:
        return None
    return input_string[article_index + len("description:") :].strip()


def extract_after_review(input_string):
    article_index = input_string.find("review:")
    if article_index == -1:
        return None
    return input_string[article_index + len("review:") :].strip()


def extract_after_paper(input_string):
    article_index = input_string.find("paper:")
    if article_index == -1:
        return None
    return input_string[article_index + len("paper:") :].strip()


def extract_after_abstract(input_string):
    article_index = input_string.find("abstract:")
    if article_index == -1:
        return None
    return input_string[article_index + len("abstract:") :].strip()


def extract_after_colon(input_string):
    article_index = input_string.find(":")
    if article_index == -1:
        return None
    return input_string[article_index + len(":") :].strip()


def add_string_after_title(original_string, string_to_add):
    title_index = original_string.find("title")

    if title_index == -1:
        return original_string

    return (
        original_string[: title_index + 5]
        + ", and "
        + string_to_add
        + original_string[title_index + 5 :]
    )


def batchify(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


###################
# Corpus Makers
###################


def classification_citation_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x["id"] for x in profile]
    extracted = extract_strings_between_quotes(inp)
    query = f"{extracted[1]} {extracted[2]}"
    return corpus, query, ids


def classification_review_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x["id"] for x in profile]
    query = extract_after_review(inp)
    return corpus, query, ids


def generation_news_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["text"]}' for x in profile]
    ids = [x["id"] for x in profile]
    query = extract_after_article(inp)
    return corpus, query, ids


def generation_paper_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["title"]} {x["abstract"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["title"]} {x["abstract"]}' for x in profile]
    ids = [x["id"] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids


def parphrase_tweet_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    query = extract_after_colon(inp)
    ids = [x["id"] for x in profile]
    return corpus, query, ids


def generation_avocado_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["text"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["text"]}' for x in profile]
    ids = [x["id"] for x in profile]
    query = extract_after_colon(inp)
    return corpus, query, ids


def classification_movies_query_corpus_maker(inp, profile, use_date):
    if use_date:
        corpus = [f'{x["description"]} date: {x["date"]}' for x in profile]
    else:
        corpus = [f'{x["description"]}' for x in profile]
    query = extract_after_description(inp)
    ids = [x["id"] for x in profile]
    return corpus, query, ids


###################
# Retrieval Related
###################


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def retrieve_top_k_with_contriver(
    contriver, tokenizer, corpus, profile, query, k, batch_size=16
) -> list[tuple]:
    """
    Returns list of tuples.
    Each tuple is (document, score)
    """
    query_tokens = tokenizer(
        [query], padding=True, truncation=True, return_tensors="pt"
    ).to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(
        output_query.last_hidden_state, query_tokens["attention_mask"]
    )
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    for batch in batched_corpus:
        tokens_batch = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda:0")
        outputs_batch = contriver(**tokens_batch)
        outputs_batch = mean_pooling(
            outputs_batch.last_hidden_state, tokens_batch["attention_mask"]
        )
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    topk_values = topk_values.tolist()
    topk_indices = topk_indices.tolist()
    return [(profile[i], score) for score, i in zip(topk_values, topk_indices)]


def retrieve_top_k_with_bm25(corpus, profile, query, k) -> list[tuple]:
    """
    Returns list of tuples.
    Each tuple is (document, score)
    """
    tokenized_corpus = [x.split() for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    selected_profs_with_scores = bm25.get_top_n_with_scores(
        tokenized_query, profile, n=k
    )
    return selected_profs_with_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )
    parser.add_argument(
        "--ranker",
        type=str,
        required=True,
        help="bm25; contriever",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="lamp; lamp1000_usable",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_date", action="store_true")
    parser.add_argument("--contriever_checkpoint", default="facebook/contriever")

    args = parser.parse_args()

    LAMP_NUM: int = args.lamp_num
    RANKER: str = args.ranker
    DATASET: str = args.dataset
    BATCH_SIZE: int = args.batch_size
    INPUT_DATA_FP = os.path.join(
        os.path.dirname(CUR_DIR_PATH),
        "data",
        DATASET,
        f"{LAMP_NUM}_user_dev_inputs.json",
    )
    OUTPUT_RANKING_DIR_PATH = os.path.join(
        CUR_DIR_PATH, "retrieval_results", DATASET, RANKER
    )
    os.makedirs(OUTPUT_RANKING_DIR_PATH, exist_ok=True)
    OUTPUT_RANKING_FP = os.path.join(OUTPUT_RANKING_DIR_PATH, f"{LAMP_NUM}.json")

    with open(INPUT_DATA_FP, "r") as file:
        dataset = json.load(file)

    rank_dict = dict()

    for data in tqdm(dataset):
        inp = data["input"]
        profile = data["profile"]
        if LAMP_NUM == 1:
            corpus, query, ids = classification_citation_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 2:
            corpus, query, ids = classification_movies_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 3:
            corpus, query, ids = classification_review_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 4:
            corpus, query, ids = generation_news_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 5:
            corpus, query, ids = generation_paper_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 6:
            corpus, query, ids = generation_avocado_query_corpus_maker(
                inp, profile, args.use_date
            )
        elif LAMP_NUM == 7:
            corpus, query, ids = parphrase_tweet_query_corpus_maker(
                inp, profile, args.use_date
            )
        else:
            raise Exception("LaMP number is between 1 and 7 inclusive")

        if RANKER == "contriever":
            tokenizer = AutoTokenizer.from_pretrained(args.contriever_checkpoint)
            contriver = AutoModel.from_pretrained(args.contriever_checkpoint).to(
                "cuda:0"
            )
            contriver.eval()
            randked_profile = retrieve_top_k_with_contriver(
                contriver, tokenizer, corpus, profile, query, len(profile), BATCH_SIZE
            )
        elif RANKER == "bm25":
            randked_profile = retrieve_top_k_with_bm25(
                corpus, profile, query, len(profile)
            )
        # elif ranker == "recency":
        #     profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
        #     randked_profile = profile[::-1]
        else:
            raise NotImplementedError

        data["profile"] = randked_profile
        # print(randked_profile)
        rank_dict[data["id"]] = [
            (prof_score_tup[0]["id"], prof_score_tup[1])
            for prof_score_tup in randked_profile
        ]

    with open(OUTPUT_RANKING_FP, "w") as f:
        json.dump(rank_dict, f)
        f.close()
