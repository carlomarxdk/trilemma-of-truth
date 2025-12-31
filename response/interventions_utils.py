"""Utilities for causal interventions on language model representations.

Provides data processors for formatting intervention experiments and functions
for translating hidden states along probe directions.
"""

from __future__ import annotations

import numpy as np
import torch


def random_answer_ids(
    seq_ids: list[list[torch.Tensor]], vocab_size: int
) -> list[torch.Tensor]:
    """Generate random answer token IDs for each sequence in seq_ids.

    Args:
        seq_ids: List of N sequences (each sequence is a list of token IDs).
        vocab_size: Size of the vocabulary (int).

    Returns:
        List of N tensors, each containing random token IDs of the same length
        as the corresponding sequence in seq_ids.

    """
    output = []
    for seq in seq_ids:
        rnd_seq = np.random.choice(vocab_size, len(seq), replace=False)
        output.append(torch.tensor(rnd_seq))
    return output


def compute_layer_scale(dh, direction, layer_id, eps=1e-6):
    """Compute σ_layer: the standard deviation of projection onto direction.

    Normalizes direction and projects activations onto it, then computes
    standard deviation across all positions.

    Args:
        dh: DataHandler object containing training data.
        direction: Concept direction vector (will be normalized).
        layer_id: Index of the layer to analyze.
        eps: Minimum value for clamping standard deviation.

    Returns:
        Standard deviation of projections as a float.

    """
    # Normalize direction
    unit_dir = direction / direction.norm()

    # Expected shape: [B, S, H]
    X = dh.train_bags(layer_id)["last_embedding"].to(direction.device)

    if X.ndim != 2:
        raise ValueError(f"Expected X to be [B, S, H], got {X.shape}")

    # Project onto direction → [B, S]
    coords = torch.einsum("bh, h -> b", X, unit_dir)

    # Pool over batch and tokens → scalar
    sigma = coords.flatten().std().clamp_min(eps)

    return sigma.item()


def normalize_logprob(lp: torch.Tensor) -> torch.Tensor:
    """Convert token or multi-piece log-prob tensor into a scalar.

    If lp is a vector (e.g. BPE pieces), sum its entries.

    Args:
        lp: Log-probability tensor (scalar or 1D).

    Returns:
        Scalar log-probability tensor.

    """
    if lp.ndim == 0:
        return lp
    return lp.sum()


def mean_logprobs(logprobs: list[torch.Tensor]) -> float:
    """Compute mean log-probability across answer tokens.

    Handles variable-length tokenizations by summing sub-token
    log-probabilities per token before averaging.

    Args:
        logprobs: List of log-probability tensors (one per token).

    Returns:
        Mean log-probability as a float.

    """
    token_logps = [normalize_logprob(lp) for lp in logprobs]
    return float(torch.stack(token_logps).mean())


def sum_logprobs(logprobs: list[torch.Tensor]) -> float:
    """Compute total log-likelihood of an answer sequence.

    Handles variable-length tokenizations by summing sub-token
    log-probabilities per token.

    Args:
        logprobs: List of log-probability tensors (one per token).

    Returns:
        Total log-likelihood as a float.

    """
    token_logps = [normalize_logprob(lp) for lp in logprobs]
    return float(torch.stack(token_logps).sum())


class InterventionDataProcessor:
    """Handle data formatting for intervention experiments.

    Processes test data and formats statements according to dataset-specific
    templates for causal intervention experiments.

    """

    def __init__(self, datahandler, tokenizer, datapack_name):
        """Initialize the data processor.

        Args:
            datahandler: DataHandler object.
            tokenizer: Tokenizer object.
            datapack_name: Name of the datapack (str).

        """
        self.dh = datahandler
        self.datapack = datapack_name
        self.tokenizer = tokenizer

    def template(self, object_1, object_2, negation, category=None):
        """Apply dataset-specific statement template.

        Args:
            object_1: First object (str).
            object_2: Second object (str).
            negation: Negation flag (0 or 1).
            category: Optional category for definitions dataset.

        Returns:
            Formatted statement string.

        """
        article = "is" if negation == 0 else "is not"
        if self.datapack in ["cities", "cities_loc"]:
            if "city" in object_1.lower():
                return f"{object_1} is located in"
            return f"The city of {object_1} {article} located in"
        elif self.datapack in ["drugs", "med_indications"]:
            if any(
                word in object_1.lower()
                for word in [
                    "control",
                    "preparation",
                    "contraception",
                    "prevention",
                    "weight loss",
                ]
            ):
                return (
                    f"{object_1.capitalize()} {article} indicated for the treatment of"
                )
            return f"{object_1.capitalize()} {article} indicated for the treatment of"
        elif self.datapack == "symptoms":
            return f"{object_1.capitalize()} {article} linked to"
        elif self.datapack in ["definitions", "defs"]:
            if category == "instances":  # noqa: SIM116
                return f"{object_1} {article} a"
            elif category == "synonyms":
                return f"{object_1} {article} a synonym of a"
            elif category == "types":
                return f"{object_1} {article} a type of a"
            else:
                return f"{object_1} {article} a"
        else:
            raise ValueError("Invalid data pack")

    def return_processed_test_df(self):
        """Process test data with templated statements.

        Returns:
            DataFrame with 'statement' and 'answer' columns added.

        """
        test_data = self.dh.get_test_df()[
            [
                "object_1",
                "object_2",
                "correct_object_2",
                "real_object",
                "correct",
                "negation",
                "category",
            ]
        ]

        test_data["answer"] = test_data["object_2"]
        test_data["statement"] = test_data.apply(
            lambda row: self.template(
                row["object_1"], row["object_2"], row["negation"], row["category"]
            ),
            axis=1,
        )

        return test_data

    def get_answer_ids(self, answer):
        """Tokenize answer and return token IDs.

        Args:
            answer: Answer string to tokenize.

        Returns:
            Tensor of token IDs.

        """
        return self.tokenizer(
            answer, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]

    def get_answer_seq_ids(self, statement, answer):
        """Generate incremental sequences for multi-token answers.

        Splits answer into words and creates progressive statement sequences.

        Args:
            statement: Base statement string.
            answer: Answer string (may contain multiple words).

        Returns:
            Tuple of (statements, answers, answer_ids, init_statement_ids).

        """
        answers = [" " + a.rstrip() for a in answer.split(" ")]
        answers_ids = []
        current = statement
        if current[-1] == " ":
            current = current.rstrip()
        statements = [current]
        init_statement_ids = self._statement_to_ids(current)

        for a in answers:
            current += a
            answers_ids.append(self._answer_to_ids(a))
            statements.append(current)
        return statements, answers, answers_ids, init_statement_ids

    def _statement_to_ids(self, statement):
        """Convert statement to token IDs.

        Args:
            statement: Statement string.

        Returns:
            List of token IDs.

        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(statement))

    def _answer_to_ids(self, answer):
        """Convert answer to token IDs.

        Args:
            answer: Answer string.

        Returns:
            List of token IDs.

        """
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))


class InstructInterventionDataProcessor(InterventionDataProcessor):
    """Data processor for instruction-tuned models.

    Extends InterventionDataProcessor to format statements using
    instruction templates with system/user/assistant roles.

    """

    def __init__(
        self, datahandler, tokenizer, datapack_name, user_role, system_role, assist_role
    ):
        """Initialize instruction-based data processor.

        Args:
            datahandler: DataHandler object.
            tokenizer: Tokenizer object.
            datapack_name: Name of the datapack (str).
            user_role: User role identifier.
            system_role: System role identifier.
            assist_role: Assistant role identifier.

        """
        super().__init__(datahandler, tokenizer, datapack_name)
        # Used only for the instruct template
        self.system_role = system_role
        self.user_role = user_role
        self.assist_role = assist_role

    def _instruct_template(self, statement: str):
        """Format statement with instruction template roles.

        Args:
            statement: Statement string to format.

        Returns:
            List of message dictionaries with role and content.

        """
        if self.system_role == self.user_role:
            return [
                {
                    "role": f"{self.user_role}",
                    "content": f"You are an expert in fact-checking. Complete this statement: {statement}",
                },
            ]
        else:
            return [
                {
                    "role": f"{self.system_role}",
                    "content": "You are an expert in fact-checking. Complete the statement provided by the user.",
                },
                {"role": f"{self.user_role}", "content": f"{statement}"},
            ]

    def _template(self, object_1, object_2, negation, category=None):
        """Apply dataset template and wrap with instruction format.

        Args:
            object_1: First object (str).
            object_2: Second object (str).
            negation: Negation flag (0 or 1).
            category: Optional category for definitions dataset.

        Returns:
            List of message dictionaries for instruction template.

        """
        statement = self.template(object_1, object_2, negation, category)
        return self._instruct_template(statement)

    def return_processed_test_df(self):
        """Process test data with instruction-formatted statements.

        Returns:
            DataFrame with templated instruction messages.

        """
        test_data = self.dh.get_test_df()[
            [
                "object_1",
                "object_2",
                "correct_object_2",
                "real_object",
                "correct",
                "negation",
                "category",
            ]
        ]
        test_data["answer"] = test_data["object_2"]
        test_data["statement"] = test_data.apply(
            lambda row: self._template(
                row["object_1"], row["object_2"], row["negation"], row["category"]
            ),
            axis=1,
        )
        return test_data

    def get_answer_ids(self, answer):
        """Tokenize answer and return token IDs.

        Args:
            answer: Answer string to tokenize.

        Returns:
            Tensor of token IDs.

        """
        return self.tokenizer(
            answer, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]

    def get_answer_seq_ids(self, statement, answer):
        """Generate incremental sequences for multi-token answers with instruction format.

        Args:
            statement: Instruction-formatted message list.
            answer: Answer string (may contain multiple words).

        Returns:
            Tuple of (statements, answers, answer_ids, init_statement_ids).

        """
        answers = [" " + a.rstrip() for a in answer.split(" ")]

        answers_ids = []
        current = self._statement_to_tokens(statement)
        statements = [current]
        init_statement_ids = self._statement_to_ids(statement)

        for aID, a in enumerate(answers):
            if aID == 0:
                current += a.lstrip()
            else:
                current += a
            answers_ids.append(self._answer_to_ids(a))
            statements.append(current)
        return statements, answers, answers_ids, init_statement_ids

    def _statement_to_tokens(self, statement):
        return self.tokenizer.apply_chat_template(
            statement, add_generation_prompt=True, tokenize=False
        )

    def _statement_to_ids(self, statement):
        return self.tokenizer.apply_chat_template(
            statement, add_generation_prompt=False, tokenize=True
        )


def translate_concept(
    X: torch.Tensor,
    direction: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """Translate embeddings by an additive shift along a concept direction.

    This applies a causal intervention of the form:
        X ← X + delta · d̂
    where d̂ is the unit-normalized concept direction.

    Args:
        X: Hidden states to modify, shape [B, S, H].
        direction: Concept direction vector, shape [H].
        delta: Scalar translation magnitude (e.g. ±sigma).

    Returns:
        Translated embeddings with the same shape as X.

    """
    unit_dir = direction / direction.norm()
    return X + delta * unit_dir.view(1, 1, -1)
