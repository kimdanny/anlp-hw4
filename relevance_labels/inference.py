"""
Perform inference with
1) 0 profile: baseline
2) 1 profile: augmented model
"""

import os
import sys

CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(CUR_DIR_PATH))

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
        use_retrieval: bool = False,
        model_kwargs: dict = None,
        pipeline_kwargs: dict = None,
    ):
        self.model_name = model_name
        self.use_retrieval = use_retrieval
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
        if "T5" in self.model_name:
            return PromptTemplate.from_template("""{final_prompt}""")
        elif "llama" in self.model_name:
            return PromptTemplate.from_template(
                """
                <s>[INST] <<SYS>>
                Follow the Question and answer.
                <</SYS>>
                Question:\n {final_prompt}\n\n
                Answer:
                [/INST]"""
            )
        else:
            raise NotImplementedError

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
    print(MODEL_NAME)
    LAMP_NUM: int = args.lamp_num
    EXPERIMENT_BASELINE: bool = args.experiment_baseline
    TOKENIZER = AutoTokenizer.from_pretrained(models_info[MODEL_NAME]["model_id"])
    K = 0 if EXPERIMENT_BASELINE else args.k

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
    col_names = ["qid", "pid", "answer", "target"]
    print("\t".join(col_names), flush=True)
    # LOG_WRITE_STREAM.write("\t".join(col_names))

    lamp_handler = LaMPHandler(
        split_type=args.lamp_split_type,
        tokenizer_model_name=models_info[MODEL_NAME]["model_id"],
        k=K,
    )
    qa_model = PromptLM(model_name=MODEL_NAME)
    aip_func = lamp_handler.get_aip_func(lamp_num=LAMP_NUM)

    inputs_file_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=LAMP_NUM)
    outputs_file_iterator = lamp_handler.get_outputs_file_iterator(lamp_number=LAMP_NUM)

    for i, (input_entry, output_entry) in enumerate(
        zip(inputs_file_iterator, outputs_file_iterator)
    ):
        # sampling first 1000 queries
        if i > 1000:
            break
        assert input_entry["id"] == output_entry["id"]
        entry_id: str = input_entry["id"]
        entry_question: str = input_entry["input"]
        profiles: List[dict] = input_entry["profile"]
        # gold label
        entry_target = output_entry["output"]

        if EXPERIMENT_BASELINE:
            answer = qa_model.answer_question(
                final_prompt=trim_sentence_by_token(entry_question, tokenizer=TOKENIZER)
            )
            s = "\t".join([entry_id, "-1", answer, entry_target])
            print(s, flush=True)
            # LOG_WRITE_STREAM.write(
            #     "\t".join([entry_id, "-1", entry_question, answer, entry_target])
            # )
            # LOG_WRITE_STREAM.write("\n")
        else:
            for profile in profiles:
                # augment with one profile one by one to test its relevancy (usefulness)
                final_prompt = aip_func(question=entry_question, profiles=[profile])
                final_prompt = trim_sentence_by_token(
                    final_prompt, tokenizer=TOKENIZER, use_model_max_length=True
                )
                answer = qa_model.answer_question(final_prompt=final_prompt)
                s = "\t".join([entry_id, profile["id"], answer, entry_target])
                print(s, flush=True)
                # LOG_WRITE_STREAM.write(
                #     "\t".join(
                #         [entry_id, profile["id"], entry_question, answer, entry_target]
                #     )
                # )
                # LOG_WRITE_STREAM.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # meta-llama/Llama-2-7b-chat-hf
    parser.add_argument(
        "--model_name",
        type=str,
        default="flanT5XXL",
        help="Model nickname of HF model",
    )

    parser.add_argument(
        "--lamp_split_type",
        type=str,
        default="user",
        help="data split type of LaMP: either 'user' or 'time'",
    )

    parser.add_argument(
        "--lamp_num",
        type=int,
        help="LaMP number",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="number of k (how many to retrieve)",
    )

    parser.add_argument(
        "--experiment_baseline",
        action="store_true",
        help="Enable baseline experiment (no profile injection)",
    )

    args = parser.parse_args()

    main(args)
