import torch
import logging
from typing import List, Dict, Union
from collections import Counter

from abc import ABC, abstractmethod
from response.prompt_templates import PromptTemplate
from transformers import AutoTokenizer
log = logging.getLogger("logit_collector")

BINARY_TRUE = ["true", "correct", "1", "yes", "right"]
BINARY_FALSE = ["false", "incorrect", "0", "no", "wrong"]

MULTICHOICE = ["1", "2", "3", "4"]

LEGAL_AGG = ["sum", "mean", "max"]
LEGAL_QT = ["binary", "binary_true", "multichoice"]


class LogitCollectorTemplate(ABC):
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 prompt_template: PromptTemplate,
                 agg: str = "sum") -> None:
        assert agg in LEGAL_AGG, f"Only {LEGAL_AGG} aggregations are supported"
        self._check_prompt_template(prompt_template)

        # Variables
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.model_name = self.tokenizer.name_or_path  # type: ignore
        self.question_type = self.prompt_template.question_type
        self.aggregation = agg
        self.enum_list = self.prompt_template.enumeration

        self.tokens = self.augment_token_list(self.enum_list)
        self.ids = self.return_token_ids(self.tokens)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return self.collect_proba(logits)

    def encode(self, token):
        return self.tokenizer.encode(token)

    def decode(self, token):
        return self.tokenizer.decode(token)

    @abstractmethod
    def _check_prompt_template(self, prompt_template: PromptTemplate) -> None:
        raise NotImplementedError()

    @abstractmethod
    def augment_token_list(self, tokens: List[str]) -> Union[List[str], Dict[str, List[str]]]:
        raise NotImplementedError()

    @abstractmethod
    def return_token_ids(self, tokens: Union[List[str], Dict[str, List[str]]]) -> Union[List[int], Dict[str, List[int]]]:
        raise NotImplementedError()

    @abstractmethod
    def collect_logits(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def collect_proba(self, logits: torch.Tensor) -> torch.Tensor:
        output = self.collect_logits(logits)
        assert output.shape[1] == len(
            self.ids) + 1, "Output shape is incorrect"
        assert output.sum(-1).allclose(torch.softmax(logits, dim=-1).sum(-1),
                                       atol=0.01), "Sum of logits does not match the sum of the collected logits"
        return output

    def collect_topn(self, logits: torch.Tensor, n: int = 1) -> List[str]:
        output = torch.softmax(logits, dim=-1)
        return [self.decode(t) for t in torch.topk(output, n, dim=-1).indices]


class MultichoiceLogitCollector(LogitCollectorTemplate):
    def _check_prompt_template(self, prompt_template):
        assert prompt_template.question_type == "multichoice", "Prompt template must be for multichoice question type"

    def augment_token_list(self, tokens: List[str]) -> Dict[str, List[str]]:
        result = {}
        for t in tokens:
            _r = [t, f" {t}"]
            result[t] = _r
        return result

    def _check_token_ids(self, tokens: Dict[str, List[Union[str, int]]]):
        # Step 1: Collect all unique tokens across all lists
        token_counts = Counter()
        for _tokens in tokens.values():
            _unique_tokens = set()  # Track unique tokens within the current list
            for t in _tokens:
                encoded = self.encode(t)  # Encode the token
                _unique_tokens.update(encoded)  # Add encoded tokens to the set

            # Update Counter with unique tokens only
            token_counts.update(_unique_tokens)

        # Step 2: Identify shared tokens (those appearing in more than one list)
        shared_tokens = {token for token,
                         count in token_counts.items() if count > 1}
        return shared_tokens

    def return_token_ids(self, tokens: Dict[str, List[int]]):
        shared_tokens = self._check_token_ids(tokens)  # type: ignore
        # Step 3: Filter out shared tokens from each list
        output = dict()
        for name, tokens in tokens.items():
            _temp = []
            for t in tokens:
                encoded_tokens = self.encode(t)
                # Keep only tokens that are not shared
                _temp.extend(
                    [tok for tok in encoded_tokens if tok not in shared_tokens])
            # Remove duplicates within the list
            output[name] = list(set(_temp))
        return output

    def collect_logits(self, logits):
        assert logits.dim() == 2, "Logits must be 2D tensor"
        logits = torch.softmax(logits, dim=-1)
        output = []
        if self.aggregation == "sum":
            _else = logits.sum(-1)
            for v in self.ids.values():
                _logit = 0
                for vv in v:
                    _logit += logits[:, vv]
                _else -= _logit
                output.append(_logit)
            output.append(_else)
            output = torch.vstack(output).T
            assert output.shape[1] == len(
                self.ids) + 1, "Output shape is incorrect"
        return output

    def collect_logits_unsafe(self, logits):
        '''
        Collect logits for NNSight without any checks
        '''
        logits = torch.softmax(logits, dim=-1)
        output = []
        if self.aggregation == "sum":
            _else = logits.sum(-1)
            for v in self.ids.values():
                _logit = 0
                for vv in v:
                    _logit += logits[:, vv]
                _else -= _logit
                output.append(_logit)
            output.append(_else)
            output = torch.vstack(output).T
        return output
