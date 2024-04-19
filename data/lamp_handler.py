# LaMP paper: https://arxiv.org/abs/2304.11406

import json
import os
from typing import List
from sys import exit
from data.data_utils import wget_file_to_dir
from transformers import AutoTokenizer


CUR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


class LaMPHandler:
    def __init__(
        self,
        lamp_dir_name: str = "lamp",
        split_type: str = "user",
        tokenizer_model_name=None,
        k: int = 1,
    ) -> None:
        self.LAMP_DIR_PATH = os.path.join(CUR_DIR_PATH, lamp_dir_name)
        self.split_type: str = split_type
        if tokenizer_model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
            self.TOKENIZER_MAX_LENGTH = self.tokenizer.model_max_length

            # For trimming ppep: following Appendix D in the LaMP paper
            # l_bar is set to 256 in LaMP
            # l is 512 for t5xxl
            l_bar = 256
            if k > 0:
                self.max_token_length_per_ppep = int(
                    (self.TOKENIZER_MAX_LENGTH - l_bar) // k
                )
            elif k == 0:
                # shouldn't reach
                self.max_token_length_per_ppep = l_bar
            else:
                raise Exception("k (int) should be non-negative")

    def __download_dev_dataset(self, user_based=False, time_based=False):
        """
        Download LaMP dev dataset to 'datasets/lamp' directory
        """
        if input("Confirm 'yes' for downloading: ") != "yes":
            exit(1)

        if user_based is False and time_based is False:
            raise Exception("Indicate which data split you want to download")

        base_url = "https://ciir.cs.umass.edu/downloads/LaMP/"
        for i in range(1, 8):
            if user_based:
                dir_url = (
                    f"{base_url}LaMP_{i}/new/dev/"
                    if i == 2
                    else f"{base_url}LaMP_{i}/dev/"
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_questions.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_user_dev_inputs.json",
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_outputs.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_user_dev_outputs.json",
                )
            if time_based:
                dir_url = (
                    f"{base_url}time/LaMP_{i}/new/dev/"
                    if i == 2
                    else f"{base_url}time/LaMP_{i}/dev/"
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_questions.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_time_dev_inputs.json",
                )
                wget_file_to_dir(
                    url=f"{dir_url}dev_outputs.json",
                    download_path=self.LAMP_DIR_PATH,
                    custom_file_name=f"{i}_time_dev_outputs.json",
                )

    def _trim_ppep(self, ppep: str) -> str:
        tokenized_ppep = self.tokenizer.tokenize(ppep)
        if len(tokenized_ppep) > self.max_token_length_per_ppep:
            truncated_tokens = tokenized_ppep[: self.max_token_length_per_ppep]
            truncated_ppep = self.tokenizer.convert_tokens_to_string(truncated_tokens)
            return truncated_ppep
        else:
            return ppep

    def _lamp_1_ppep(self, title: str) -> str:
        return self._trim_ppep(title)

    def _lamp_2_ppep(self, description: str, tag: str) -> str:
        return self._trim_ppep(f"the tag for the movie: {description} is {tag}")

    def _lamp_3_ppep(self, score: str, text: str) -> str:
        return self._trim_ppep(f"{score} is the score for {text}")

    def _lamp_4_ppep(self, title: str, text: str) -> str:
        return self._trim_ppep(f"{title} is the title for {text}")

    def _lamp_5_ppep(self, title: str, abstract: str) -> str:
        return self._trim_ppep(f"{title} is the title for {abstract}")

    def _lamp_6_ppep(self, title: str, text: str) -> str:
        return self._trim_ppep(f"{title} is the title for {text}")

    def _lamp_7_ppep(self, text: str) -> str:
        return self._trim_ppep(text)

    @staticmethod
    def _add_to_paper_title(question: str, titles: str) -> str:
        split_questions = question.split("which reference is related?")
        return (
            split_questions[0]
            + titles
            + ", which reference is related?"
            + split_questions[1]
        )

    def _lamp_1_aip(self, question: str, profiles: List[dict]) -> str:
        titles = ", and ".join(
            [self._lamp_1_ppep(title=profile["title"]) for profile in profiles]
        )
        return self._add_to_paper_title(question, titles)

    def _lamp_2_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [
                self._lamp_2_ppep(
                    description=profile["description"], tag=profile["tag"]
                )
                for profile in profiles
            ]
        )
        aip += f". {question}"
        return aip

    def _lamp_3_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [
                self._lamp_3_ppep(score=profile["score"], text=profile["text"])
                for profile in profiles
            ]
        )
        aip += f". {question}"
        return aip

    def _lamp_4_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [
                self._lamp_4_ppep(title=profile["title"], text=profile["text"])
                for profile in profiles
            ]
        )
        aip += f". {question}"
        return aip

    def _lamp_5_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [
                self._lamp_5_ppep(title=profile["title"], abstract=profile["abstract"])
                for profile in profiles
            ]
        )
        aip += f". Following the given patterns {question}"
        return aip

    def _lamp_6_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [
                self._lamp_6_ppep(title=profile["title"], text=profile["text"])
                for profile in profiles
            ]
        )
        aip += f". {question}"
        return aip

    def _lamp_7_aip(self, question: str, profiles: List[dict]) -> str:
        aip = ", and ".join(
            [self._lamp_7_ppep(text=profile["text"]) for profile in profiles]
        )
        aip += f" are written by a person. Following the given patterns {question}"
        return aip

    def get_aip_func(self, lamp_num: int):
        if lamp_num == 1:
            return self._lamp_1_aip
        elif lamp_num == 2:
            return self._lamp_2_aip
        elif lamp_num == 3:
            return self._lamp_3_aip
        elif lamp_num == 4:
            return self._lamp_4_aip
        elif lamp_num == 5:
            return self._lamp_5_aip
        elif lamp_num == 6:
            return self._lamp_6_aip
        elif lamp_num == 7:
            return self._lamp_7_aip
        else:
            return NotImplementedError

    def get_inputs_file_iterator(self, lamp_number: int):
        assert 0 < lamp_number < 8
        with open(
            os.path.join(
                self.LAMP_DIR_PATH, f"{lamp_number}_{self.split_type}_dev_inputs.json"
            ),
            "r",
        ) as f:
            data: dict = json.load(f)
        f.close()
        return iter(data)

    def get_outputs_file_iterator(self, lamp_number: int):
        assert 0 < lamp_number < 8
        with open(
            os.path.join(
                self.LAMP_DIR_PATH, f"{lamp_number}_{self.split_type}_dev_outputs.json"
            ),
            "r",
        ) as f:
            data: dict = json.load(f)
        f.close()
        return iter(data["golds"])


# if __name__ == "__main__":
#     lamp_handler = LaMPHandler(tokenizer_model_name="google/flan-t5-xxl")
#     entry_iterator = lamp_handler.get_inputs_file_iterator(lamp_number=6)
#     aip_func = lamp_handler.get_aip_func(lamp_num=6)

#     for _ in range(44):
#         next(entry_iterator)

#     input_entry = next(entry_iterator)

#     entry_id: str = input_entry["id"]
#     entry_question: str = input_entry["input"]
#     profiles: List[dict] = input_entry["profile"]
#     prompt = aip_func(entry_question, profiles=[profiles[1]])
#     print(prompt)
