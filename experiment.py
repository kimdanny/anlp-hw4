"""
Perform inference with
1) 0 profile: baseline
2) 1 profile: augmented model
"""

import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

import json
import argparse
from typing import List
import torch
from transformers import AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from utils import models_info, trim_sentence_by_token
from data.lamp_handler import LaMPHandler


class PromptLM:
    """
    model_name (str): model nickname. Can find from utils.complete_model_names
    use_retrieval (bool): Flag indicating whether retrieval-based prompting is used.
    pipeline_kwargs (Dict): Additional arguments to configure the Hugging Face pipeline.
    hf_pipeline (HuggingFacePipeline): The initialized Hugging Face pipeline for text generation.
    prompt (PromptTemplate): The template used to format prompts for the model.
    chain (PromptTemplate | HuggingFacePipeline): The chain of operations to be executed for text generation
    """

    def __init__(
        self,
        model_name: str,
        model_kwargs: dict = None,
        pipeline_kwargs: dict = None,
    ):
        self.model_name = model_name
        self.model_kwargs: dict = model_kwargs or {}
        self.pipeline_kwargs = pipeline_kwargs or {
            "max_new_tokens": 128,
            "num_beams": 4,
        }
        self.prompt = self._choose_prompt_template()
        self.hf_pipeline = self._initialize_pipeline()
        # langchain
        self.chain = self.prompt | self.hf_pipeline

    def _choose_prompt_template(self) -> PromptTemplate:
        return PromptTemplate.from_template("""{final_prompt}""")

    def _initialize_pipeline(self) -> HuggingFacePipeline:
        torch.cuda.empty_cache()
        return HuggingFacePipeline.from_model_id(
            model_id=models_info[self.model_name]["model_id"],
            task=models_info[self.model_name]["hf_pipeline_task"],
            device=0,
            model_kwargs=self.model_kwargs,
            pipeline_kwargs=self.pipeline_kwargs,
        )

    def answer_question(self, final_prompt: str) -> str:
        # torch.cuda.empty_cache()
        return self.chain.invoke({"final_prompt": final_prompt}).strip()


def main(args):
    MODEL_NAME: str = args.model_name
    LAMP_NUM: int = args.lamp_num
    DATASET: str = args.dataset
    EXPERIMENT_BASELINE: bool = args.experiment_baseline
    RETRIEVER: str = args.retriever
    TOKENIZER = AutoTokenizer.from_pretrained(models_info[MODEL_NAME]["model_id"])
    K = 0 if EXPERIMENT_BASELINE else args.k
    ADAPTIVE_K: bool = args.adaptive_k
    ADAPTIVE_M: bool = args.adaptive_M
    ret_result_fp = (
        f"{DATASET}_adaptive_k/{RETRIEVER}/LaMP{LAMP_NUM}_k{args.k}.json"
        if ADAPTIVE_K
        else f"{DATASET}/{RETRIEVER}/{LAMP_NUM}.json"
    )
    RETRIEVAL_RESULTS_FP = os.path.join(
        CUR_DIR_PATH, "retrieval", "retrieval_results", ret_result_fp
    )

    with open(RETRIEVAL_RESULTS_FP, "r") as f:
        retrieval_result_dict: dict = json.load(f)
    f.close()

    # LOG_DIR_PATH = os.path.join(CUR_DIR_PATH, "inference_results", "lamp")
    # if not os.path.exists(LOG_DIR_PATH):
    #     os.makedirs(LOG_DIR_PATH, exist_ok=True)
    # if EXPERIMENT_BASELINE:
    #     LOG_FP: str = os.path.join(
    #         LOG_DIR_PATH, "baseline", f"lamp{LAMP_NUM}_{MODEL_NAME}.log"
    #     )
    # else:
    #     LOG_FP: str = os.path.join(
    #         LOG_DIR_PATH, "augment", f"lamp{LAMP_NUM}_{MODEL_NAME}.log"
    #     )

    # LOG_WRITE_STREAM = open(LOG_FP, "a")
    # qid: question ID
    # pid: profile ID
    col_names = ["qid", "answer", "target"]
    print("\t".join(col_names), flush=True)
    # LOG_WRITE_STREAM.write("\t".join(col_names))

    lamp_handler = LaMPHandler(
        lamp_dir_name="lamp1000_usable",
        split_type=args.lamp_split_type,
        tokenizer_model_name=models_info[MODEL_NAME]["model_id"],
        k=K,
        adaptive_M=ADAPTIVE_M,
    )
    qa_model = PromptLM(model_name=MODEL_NAME)
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)

    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)

    for input_entry, output_entry in zip(inputs_file_iterator, outputs_file_iterator):
        assert input_entry["id"] == output_entry["id"]
        entry_id: str = input_entry["id"]  # qid
        entry_question: str = input_entry["input"]
        # gold label
        entry_target = output_entry["output"]

        if EXPERIMENT_BASELINE:
            answer = qa_model.answer_question(
                final_prompt=trim_sentence_by_token(entry_question, tokenizer=TOKENIZER)
            )
            s = "\t".join([entry_id, answer, entry_target])
            print(s, flush=True)
            # LOG_WRITE_STREAM.write(
            #     "\t".join([entry_id, "-1", entry_question, answer, entry_target])
            # )
            # LOG_WRITE_STREAM.write("\n")
        else:
            # Read retrieval results and get list of profiles (comlete {} format from dataset)
            top_profiles: list = retrieval_result_dict[entry_id]
            if len(top_profiles) == 0:
                # same as baseline
                answer = qa_model.answer_question(
                    final_prompt=trim_sentence_by_token(
                        entry_question, tokenizer=TOKENIZER
                    )
                )
                s = "\t".join([entry_id, answer, entry_target])
                print(s, flush=True)
            else:
                # augmented with retrieved profiles
                pids: list = [x[0] for x in top_profiles]
                selected_profiles: list[dict] = lamp_handler.find_profiles_by_pids(
                    LAMP_NUM, entry_id, pids
                )

                final_prompt = aip_func(
                    question=entry_question, profiles=selected_profiles
                )
                final_prompt = trim_sentence_by_token(
                    final_prompt, tokenizer=TOKENIZER, use_model_max_length=True
                )
                answer = qa_model.answer_question(final_prompt=final_prompt)
                s = "\t".join([entry_id, answer, entry_target])
                print(s, flush=True)
                # LOG_WRITE_STREAM.write(
                #     "\t".join(
                #         [entry_id, profile["id"], entry_question, answer, entry_target]
                #     )
                # )
                # LOG_WRITE_STREAM.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="flanT5Base",
        help="Model nickname of HF model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lamp1000_usable",
        help="lamp; lamp1000_usable",
    )
    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        help="retriever name",
    )
    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="how many to retrieve",
    )
    parser.add_argument(
        "--adaptive_k",
        action="store_true",
        help="make the k adaptive",
    )
    parser.add_argument(
        "--adaptive_M",
        action="store_true",
        help="make the context size per document adaptive to the number of document we pass in",
    )
    parser.add_argument(
        "--experiment_baseline",
        action="store_true",
        help="Enable baseline experiment (no profile injection)",
    )
    args = parser.parse_args()

    main(args)
