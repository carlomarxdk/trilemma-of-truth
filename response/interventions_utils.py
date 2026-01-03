"""Utilities for causal interventions on language model representations.

Provides data processors for formatting intervention experiments and functions
for translating hidden states along probe directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import torch
from scipy.stats import binomtest

################################
#### Intervention utilities ####
################################


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


#############################
####       Tests.        ####
#############################
def diff_of_diff_ols(
    diff_pos: np.ndarray,
    diff_neg: np.ndarray,
    diff_rand_pos: np.ndarray,
    diff_rand_neg: np.ndarray,
    dataset: pd.DataFrame,
):
    """Difference-in-differences analysis for intervention effects on token probabilities.

    Tests whether translating hidden states along a direction vector differentially
    affects correct answer tokens vs random control tokens. Uses a 2×2 factorial
    design (token type × intervention direction) with standard errors clustered
    by statement.

    Args:
        diff_pos: Change in log-probability from baseline under positive
            intervention (RES_pos - RES_orig) for correct tokens.
        diff_neg: Change in log-probability from baseline under negative
            intervention (RES_neg - RES_orig) for correct tokens.
        diff_rand_pos: Change in log-probability from baseline under positive
            intervention for random control tokens.
        diff_rand_neg: Change in log-probability from baseline under negative
            intervention for random control tokens.
        dataset: DataFrame containing 'real_object' and 'correct' columns for
            filtering to real, true statements.

    Returns:
        Fitted OLS model with clustered standard errors. Key coefficient is
        'is_correct_token:is_pos_translation' (the DiD estimator).

    Note:
        The interaction term tests:
            H0: (effect_pos - effect_neg)_correct = (effect_pos - effect_neg)_random
            H1: Positive translation boosts correct tokens more than random tokens

        Coefficients:
            - Intercept: baseline effect for random tokens under negative intervention
            - is_correct_token: main effect of token type (correct vs random)
            - is_pos_translation: main effect of intervention direction
            - is_correct_token:is_pos_translation: DiD estimator (key result)

    """
    N = diff_pos.shape[0]
    r = dataset["real_object"].values[:N]
    c = dataset["correct"].values[:N]

    mask = (r == 1) & (c == 1)
    M = mask.sum()

    df = pd.DataFrame(
        {
            "effect": np.concatenate(
                [
                    diff_pos[mask],
                    diff_neg[mask],
                    diff_rand_pos[mask],
                    diff_rand_neg[mask],
                ]
            ),
            # More interpretable names
            "is_correct_token": ([1] * M) + ([1] * M) + ([0] * M) + ([0] * M),
            "is_pos_translation": ([1] * M) + ([0] * M) + ([1] * M) + ([0] * M),
            "statement": np.tile(np.arange(M), 4),
        }
    )

    y, X = patsy.dmatrices(
        "effect ~ is_correct_token * is_pos_translation",
        data=df,
        return_type="dataframe",
    )
    groups = df.loc[X.index, "statement"].to_numpy()
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    return model


def intervention_success_rate(
    diff_pos: np.ndarray,
    diff_neg: np.ndarray,
    eps: float = 1e-12,
    dataset: pd.DataFrame | None = None,
):
    """
    Compute per-statement intervention success based on directional consistency.

    A statement is considered successful if:
      (1) Positive and negative interventions have opposing effects, and
      (2) The positive intervention aligns with the dominant direction.

    Args:
        diff_pos: Array of per-statement effects under positive intervention
                  (e.g., RES_pos - RES_orig).
        diff_neg: Array of per-statement effects under negative intervention
                  (e.g., RES_neg - RES_orig).
        eps: Small tolerance to treat near-zero effects as zero.
        dataset: Optional DataFrame with 'real_object' and 'correct' columns
                 for filtering to real, true statements.

    Returns:
        dict with:
            - success_rate: ω, fraction of successful statements
            - dominant_direction: +1 or -1
            - n_success: number of successful statements
            - n_total: number of valid statements
            - p_value: one-sided binomial test p-value (H0: ω ≤ 0.5)
            - opposition_rate: fraction with sign(diff_pos) != sign(diff_neg)
    """
    assert diff_pos.shape == diff_neg.shape, "diff_pos and diff_neg must match"

    if dataset is not None:
        N = diff_pos.shape[0]
        r = dataset["real_object"].values[:N]
        c = dataset["correct"].values[:N]

        mask = (r == 1) & (c == 1)
        diff_pos = diff_pos[mask]
        diff_neg = diff_neg[mask]

    # Treat near-zero effects as zero
    dp = np.where(np.abs(diff_pos) < eps, 0.0, diff_pos)
    dn = np.where(np.abs(diff_neg) < eps, 0.0, diff_neg)

    sign_pos = np.sign(dp)
    sign_neg = np.sign(dn)

    # Keep only statements with non-zero effects on both sides
    valid = (sign_pos != 0) & (sign_neg != 0)
    if valid.sum() == 0:
        return {
            "success_rate": np.nan,
            "dominant_direction": np.nan,
            "n_success": 0,
            "n_total": 0,
            "p_value": np.nan,
            "opposition_rate": np.nan,
        }

    sign_pos = sign_pos[valid]
    sign_neg = sign_neg[valid]

    # (1) Opposing effects
    opposition = sign_pos != sign_neg
    opposition_rate = opposition.mean()

    # Dominant direction of positive intervention
    dominant_direction = np.sign(sign_pos.sum())
    if dominant_direction == 0:
        # Perfect tie → no dominant direction
        dominant_direction = np.nan

    # (2) Alignment with dominant direction
    aligned = sign_pos == dominant_direction

    # Success definition
    success = opposition & aligned
    n_success = success.sum()
    n_total = success.size
    success_rate = n_success / n_total

    # One-sided binomial test: H0: ω ≤ 0.5, H1: ω > 0.5
    p_value = binomtest(n_success, n_total, p=0.5, alternative="greater").pvalue

    return {
        "success_rate": float(success_rate),
        "dominant_direction": float(dominant_direction),
        "n_success": int(n_success),
        "n_total": int(n_total),
        "p_value": float(p_value),
        "opposition_rate": float(opposition_rate),
    }


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
        if self.datapack in ["city_locations", "cities_loc"]:
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
        elif self.datapack in ["word_definitions", "defs"]:
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
