# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py

import os
import pandas as pd
import argparse
import ast
import matplotlib.pyplot as plt

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


from eval.lamp_metrics import (
    get_metric_fn_accuracy,
    get_metric_fn_f1,
    get_metric_fn_mae,
    # get_metric_fn_rmse,
    # get_metric_fn_rouge_1,
    get_metric_fn_rouge_L,
)


def load_df(fp: str) -> pd.DataFrame:
    # Must read qid as string not as int
    dtype_spec = {"qid": str, "answer": str, "target": str}
    df = pd.read_csv(fp, delimiter="\t", dtype=dtype_spec)
    return df


def get_labels(lamp_num):
    if lamp_num == 1:
        return ["[1]", "[2]"]
    elif lamp_num == 2:
        return [
            "sci-fi",
            "based on a book",
            "comedy",
            "action",
            "twist ending",
            "dystopia",
            "dark comedy",
            "classic",
            "psychology",
            "fantasy",
            "romance",
            "thought-provoking",
            "social commentary",
            "violence",
            "true story",
        ]
    elif lamp_num == 3:
        return ["1", "2", "3", "4", "5"]
    else:
        raise ValueError(f"LaMP {lamp_num} is not classification task")


def read_string_to_list(string_list: str) -> list:
    res = ast.literal_eval(string_list)
    res = [float(x) for x in res]
    return res


def main(args):
    LAMP_NUM: int = args.lamp_num

    # set corresponding metric function for a LaMP task
    if LAMP_NUM == 1:
        metric_fn = get_metric_fn_accuracy(get_labels(LAMP_NUM))
        metric_name = "Accuracy"
    elif LAMP_NUM == 2:
        metric_fn = get_metric_fn_f1(get_labels(LAMP_NUM))
        metric_name = "F1"
    elif LAMP_NUM == 3:
        metric_fn = get_metric_fn_mae()
        metric_name = "MAE"
    else:
        metric_fn = get_metric_fn_rouge_L()
        metric_name = "ROUGE-L"

    for RETRIEVER in ["bm25", "contriever"]:
        for ADAPTIVE_K in [True, False]:
            retriever_dirname: str = (
                f"{RETRIEVER}_adaptive" if ADAPTIVE_K else f"{RETRIEVER}_topk"
            )
            EVAL_RESULTS_DIR_PATH = os.path.join(
                CUR_DIR_PATH, "plots", retriever_dirname
            )
            os.makedirs(EVAL_RESULTS_DIR_PATH, exist_ok=True)

            scores_for_ks = []
            for K in range(6):
                INF_RESULTS_FP = os.path.join(
                    CUR_DIR_PATH,
                    "exp_outputs",
                    retriever_dirname,
                    f"lamp_{LAMP_NUM}_k_{K}.log",
                )
                # get average metric score
                inf_df = load_df(os.path.join(INF_RESULTS_FP))
                inf_answers = inf_df["answer"].tolist()
                inf_answers = [str(x) for x in inf_answers]
                inf_targets = inf_df["target"].tolist()
                inf_targets = [str(x) for x in inf_targets]
                inf_scores: list = metric_fn(inf_answers, inf_targets)
                score_avg: float = sum(inf_scores) / len(inf_scores)
                scores_for_ks.append(score_avg)

            # Save scores for Ks in txt file
            with open(
                os.path.join(EVAL_RESULTS_DIR_PATH, f"lamp_{LAMP_NUM}_scores.txt"), "w"
            ) as f:
                rounded_scores = [round(x, 4) for x in scores_for_ks]
                f.write(str(rounded_scores))
                f.write("\n")
                f.write(f"baseline: {rounded_scores[0]}\n")
                f.write(f"max: {max(rounded_scores[1:])}")
            f.close()

    # Plot scores
    for RETRIEVER in ["bm25", "contriever"]:
        retriever_dir = os.path.join(CUR_DIR_PATH, "plots", RETRIEVER)
        os.makedirs(retriever_dir, exist_ok=True)

        with open(
            os.path.join(
                CUR_DIR_PATH,
                "plots",
                f"{RETRIEVER}_adaptive/lamp_{LAMP_NUM}_scores.txt",
            ),
            "r",
        ) as f:
            adaptive_scores = f.readline().strip()
            adaptive_scores: list = read_string_to_list(adaptive_scores)
        f.close()
        with open(
            os.path.join(
                CUR_DIR_PATH, "plots", f"{RETRIEVER}_topk/lamp_{LAMP_NUM}_scores.txt"
            ),
            "r",
        ) as f:
            topk_scores = f.readline().strip()
            topk_scores: list = read_string_to_list(topk_scores)
        f.close()

        # plotting
        xs = [x for x in range(6)]
        plt.plot(xs, adaptive_scores, label="Adaptive-k", marker="o")
        plt.plot(xs, topk_scores, label="Top-k", marker="o")
        plt.legend(loc="upper right")
        plt.xlabel("k")
        plt.ylabel(metric_name)
        plt.title(f"LaMP {LAMP_NUM} with {RETRIEVER}")
        plt.savefig(os.path.join(retriever_dir, f"{LAMP_NUM}.png"))
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lamp_num",
        type=int,
        required=True,
        help="LaMP number",
    )
    # parser.add_argument(
    #     "--retriever",
    #     type=str,
    #     required=True,
    #     help="retriever name",
    # )
    # parser.add_argument(
    #     "--adaptive_k",
    #     action="store_true",
    #     help="make the k adaptive",
    # )

    args = parser.parse_args()

    main(args)
