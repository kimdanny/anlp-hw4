import argparse
from typing import List
import os
import json

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_adaptive_threshold(list_of_scores: List[List[float]], k: float):
    """
    Parameters
    list_of_scores:
        Example. Q1's retrieved docs' scores: [4, 3, 1], Q2's scores: [3, 3, 2]
        then, it should be [ [4, 3, 1], [3, 3, 2] ]
        Fyi, it doesn't need to be sorted
    k: hyperparameter (float)

    Returns
    threshold (float): it has to
    """
    assert k > 0
    num_questions = len(list_of_scores)
    scores = [s for ss in list_of_scores for s in ss]
    scores = sorted(scores, reverse=True)
    thr = scores[int(num_questions * k) - 1]
    return thr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="lamp1000_usable",
        help="lamp; lamp1000_usable",
    )
    parser.add_argument(
        "--ranker",
        type=str,
        required=True,
        help="bm25; contriever",
    )
    parser.add_argument(
        "--lamp_num",
        type=int,
        required=True,
        help="1 <= LaMP number <= 7",
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="how many to retrieve on average",
    )

    args = parser.parse_args()

    DATASET: str = args.dataset
    RANKER: str = args.ranker
    LAMP_NUM: int = args.lamp_num
    K: int = args.k
    RETRIEVAL_RESULT_FP = os.path.join(
        CUR_DIR_PATH, "retrieval_results", DATASET, RANKER, f"{LAMP_NUM}.json"
    )
    ADAPTIVE_K_RESULT_DIR_PATH = os.path.join(
        CUR_DIR_PATH, "retrieval_results", f"{DATASET}_adaptive_k", RANKER
    )
    os.makedirs(ADAPTIVE_K_RESULT_DIR_PATH, exist_ok=True)
    ADAPTIVE_K_RESULT_FP = os.path.join(
        ADAPTIVE_K_RESULT_DIR_PATH, f"LaMP{LAMP_NUM}_k{K}.json"
    )

    with open(RETRIEVAL_RESULT_FP, "r") as f:
        retrieval_result_dict: dict = json.load(f)
    f.close()

    # Getting the score threshold for the lamp task
    list_of_scores_list = []
    for qid in retrieval_result_dict:
        pid_score_list = retrieval_result_dict[qid]
        score_list = [x[1] for x in pid_score_list]
        list_of_scores_list.append(score_list)

    score_threshold = get_adaptive_threshold(list_of_scores_list, k=K)
    print(f"threshold: {score_threshold}")

    # Cutoff with the threshold
    new_rank_dict = dict()
    for qid in retrieval_result_dict:
        pid_score_list = retrieval_result_dict[qid]
        cut_off_score_list = [x for x in pid_score_list if x[1] >= score_threshold]
        new_rank_dict.update({qid: cut_off_score_list})

    with open(ADAPTIVE_K_RESULT_FP, "w") as f:
        json.dump(new_rank_dict, f)
        f.close()
