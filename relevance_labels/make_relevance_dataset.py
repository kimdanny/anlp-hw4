"""
1. Filter out the original dataset to have only usable queries (more than one relevant document per query)
2. Make qid-pid-relevance label mapping file (criteria: 1 if delta > 0 else 0)
3. Make new dataset consist of only usable queries
"""

import pandas as pd
import argparse
import os
import sys
from tqdm import tqdm
import json


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))
from data.lamp_handler import LaMPHandler


def get_stat_usable_df(df, return_latex_table_line=True) -> dict:
    stat_df = df.describe()
    res = {
        "# of queries": len(df),
        "Avg # docs": stat_df["total_entries"]["mean"],
        "Std # docs": stat_df["total_entries"]["std"],
        "Avg # of positive labels per query": stat_df["positive_deltas"]["mean"],
        "Std # of positive labels per query": stat_df["positive_deltas"]["std"],
        "Avg of (# positive labels / # total documents) ": stat_df["positive_ratio"][
            "mean"
        ],
    }
    if not return_latex_table_line:
        return res
    return " & ".join([str(round(x, 2)) for x in res.values()])


def main(args, print_stat=False):
    LAMP_NUM = args.lamp_num
    SPLIT_TYPE = args.lamp_split_type
    MODEL_NAME = args.model_name
    DATASET = args.dataset
    DELTA_ANAL_FP = os.path.join(
        CUR_DIR_PATH,
        "eval_results",
        MODEL_NAME,
        DATASET,
        f"{LAMP_NUM}_delta_analysis.tsv",
    )
    DELTA_FP = os.path.join(
        CUR_DIR_PATH,
        "eval_results",
        MODEL_NAME,
        DATASET,
        f"{LAMP_NUM}_delta.tsv",
    )
    NEW_DATA_DIR_PATH = os.path.join(
        os.path.dirname(CUR_DIR_PATH), "data", f"{DATASET}_usable"
    )
    os.makedirs(NEW_DATA_DIR_PATH, exist_ok=True)

    # 1. load analysis tsv and get 1) statistics, and 2) usable qid list
    delta_anal_df = pd.read_csv(DELTA_ANAL_FP, sep="\t", dtype={"qid": str})
    usable_delta_anal_df = delta_anal_df[delta_anal_df["positive_deltas"] > 1]
    if print_stat:
        print(
            f"LaMP {LAMP_NUM}'s usable_df stat: \n",
            get_stat_usable_df(df=usable_delta_anal_df),
        )

    usable_qids = list(usable_delta_anal_df["qid"])  # list of usable qid's
    del usable_delta_anal_df, delta_anal_df

    # 2. make qid-pid-rel mapping file
    delta_df = pd.read_csv(DELTA_FP, sep="\t", dtype={"qid": str, "pid": str})
    delta_df = delta_df[["qid", "pid", "delta"]]
    usable_delta_df = delta_df[delta_df["qid"].isin(usable_qids)]
    del delta_df
    usable_delta_df["relevance_label"] = (usable_delta_df["delta"] > 0).astype(int)
    usable_delta_df = usable_delta_df.drop(columns=["delta"], axis=1)
    # save to new dataset
    usable_delta_df.to_csv(
        os.path.join(NEW_DATA_DIR_PATH, f"{LAMP_NUM}_relevance_mapping.tsv"),
        sep="\t",
        index=False,
    )

    # 3. make filtered dataset
    lamp_handler = LaMPHandler(lamp_dir_name="lamp", split_type=SPLIT_TYPE)
    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)

    usable_input_json = []
    usable_output_json = {"task": f"LaMP_{LAMP_NUM}", "golds": []}
    for input_entry, output_entry in tqdm(
        zip(inputs_file_iterator, outputs_file_iterator)
    ):
        assert input_entry["id"] == output_entry["id"]
        entry_id: str = input_entry["id"]
        if entry_id in usable_qids:
            usable_input_json.append(input_entry)
            usable_output_json["golds"].append(output_entry)

    # save
    with open(
        os.path.join(NEW_DATA_DIR_PATH, f"{LAMP_NUM}_{SPLIT_TYPE}_dev_inputs.json"), "w"
    ) as f:
        json.dump(usable_input_json, f, indent=2)
        f.close()

    with open(
        os.path.join(NEW_DATA_DIR_PATH, f"{LAMP_NUM}_{SPLIT_TYPE}_dev_outputs.json"),
        "w",
    ) as f:
        json.dump(usable_output_json, f, indent=2)
        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="flanT5XXL",
        help="Model nickname of HF model",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="lamp1000",
        help="Dataset Name",
    )

    args = parser.parse_args()

    main(args)
