"""Utilities for causal interventions on language model representations.

Provides data processors for formatting intervention experiments and functions
for translating hidden states along probe directions.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import torch
from scipy.stats import binomtest

log = logging.getLogger("InterventionUtils")

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
        Dictionary of descriptive statistics.

    """
    # Normalize direction
    unit_dir = direction / direction.norm()

    # Expected shape: [N, H]
    X = dh.cal_bags(layer_id)["last_embedding"].to(direction.device)
    if X.ndim != 2:
        raise ValueError(f"Expected X to be [N, H], got {X.shape}")
    coords = torch.einsum("nh, h -> n", X, unit_dir)

    # Pool over batch and tokens → scalar
    sigma = coords.flatten().std().clamp_min(eps)
    
    
    stats = {
            'mean': coords.mean().item(),
            'std': coords.std().item(),
            'median': coords.median().item(),
            'iqr': (coords.quantile(0.75) - coords.quantile(0.25)).item(),
            'min': coords.min().item(),
            'max': coords.max().item(),
            'n_calibration': len(coords)
        }

    return sigma.item(), stats


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
    additional_mask: np.ndarray | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
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
        additional_mask: Optional boolean array to further filter statements.

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
    if additional_mask is not None:
        mask = mask & additional_mask
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
    additional_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
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
        if additional_mask is not None:
            mask = mask & additional_mask
        diff_pos = diff_pos[mask]
        diff_neg = diff_neg[mask]
    if additional_mask is not None:
        mask &= additional_mask[:diff_pos.shape[0]]

    # Zero out numerically small effects
    dp = np.where(np.abs(diff_pos) < eps, 0.0, diff_pos)
    dn = np.where(np.abs(diff_neg) < eps, 0.0, diff_neg)

    sign_pos = np.sign(dp).astype(int)
    sign_neg = np.sign(dn).astype(int)
    
    has_zero = (sign_pos == 0) | (sign_neg == 0)

    # --- dominant direction (ignore zeros) ---
    nonzero_pos = sign_pos[sign_pos != 0]
    if nonzero_pos.size == 0:
        dominant_direction = 0
    else:
        pos_ct = np.count_nonzero(nonzero_pos == 1)
        neg_ct = np.count_nonzero(nonzero_pos == -1)
        dominant_direction = 1 if pos_ct >= neg_ct else -1
    
    aligned_dom = (~has_zero) & (sign_pos == dominant_direction)
    aligned_opp = (~has_zero) & (sign_pos == -dominant_direction)

    # Sanity: these three sets are disjoint and cover everything
    assert np.all(has_zero | aligned_dom | aligned_opp) # noqa: S101
    assert not np.any(has_zero & aligned_dom) # noqa: S101
    assert not np.any(has_zero & aligned_opp) # noqa: S101
    assert not np.any(aligned_dom & aligned_opp) # noqa: S101
     
    rate_dom = aligned_dom.mean()
    rate_opp = aligned_opp.mean()
    rate_zero = has_zero.mean()

    assert np.isclose(rate_dom + rate_opp + rate_zero, 1.0, atol=1e-12) # noqa: S101
    
    opposing = (sign_pos * sign_neg) == -1
    success = opposing & aligned_dom
    

    n_total = success.size
    n_success = success.sum()
    success_rate = n_success / n_total
    

    result = binomtest(
        n_success, n_total, p=0.5, alternative="greater"
    )

    return {
        "success_rate": float(success_rate),
        "dominant_direction": float(dominant_direction),
        "stat":  float(result.statistic),
        "n_success": int(n_success),
        "n_total": int(n_total),
        "p_value": float(result.pvalue),
        "rate_dom": float(rate_dom),
        "rate_opp": float(rate_opp),
        "rate_zero": float(rate_zero),
    }

def test_asymmetry(
    diff_pos: np.ndarray,
    diff_neg: np.ndarray,
    diff_rand_pos: np.ndarray,
    diff_rand_neg: np.ndarray,
    dataset: pd.DataFrame,
    additional_mask: np.ndarray | None = None,
) -> dict:
    """Test whether intervention effects are symmetric.
    Args:
        diff_pos: Array of per-statement effects under positive intervention
                  (e.g., RES_pos - RES_orig).
        diff_neg: Array of per-statement effects under negative intervention
                  (e.g., RES_neg - RES_orig).
        diff_rand_pos: Array of per-statement effects under positive intervention
                    for random tokens.
        diff_rand_neg: Array of per-statement effects under negative intervention
                    for random tokens.  
        dataset: DataFrame containing 'real_object' and 'correct' columns
            for filtering to real, true statements.
        additional_mask: Optional boolean array to further filter statements.
    
    Returns:
        Statistical tests for asymmetry in:
        - Correct tokens
        - Random tokens  
        - Differential asymmetry (correct vs random)
    """
    N = diff_pos.shape[0]
    r = dataset["real_object"].values[:N]
    c = dataset["correct"].values[:N]
    
    mask = (r == 1) & (c == 1)
    if additional_mask is not None:
        mask = mask & additional_mask
    
    # For correct tokens
    abs_pos_correct = np.abs(diff_pos[mask])
    abs_neg_correct = np.abs(diff_neg[mask])
    
    # For random tokens
    abs_pos_random = np.abs(diff_rand_pos[mask])
    abs_neg_random = np.abs(diff_rand_neg[mask])
    
    # Paired t-tests for absolute effects
    from scipy.stats import ttest_rel
    
    asymmetry_correct = ttest_rel(abs_pos_correct, abs_neg_correct)
    asymmetry_random = ttest_rel(abs_pos_random, abs_neg_random)
    
    # Differential asymmetry (interaction)
    diff_asymmetry_correct = abs_pos_correct - abs_neg_correct
    diff_asymmetry_random = abs_pos_random - abs_neg_random
    
    differential = ttest_rel(diff_asymmetry_correct, diff_asymmetry_random)
    
    return {
        "correct_asymmetry": {
            "mean_pos": float(abs_pos_correct.mean()),
            "mean_neg": float(abs_neg_correct.mean()),
            "ratio": float(abs_pos_correct.mean() / abs_neg_correct.mean()),
            "t_stat": float(asymmetry_correct.statistic),
            "p_value": float(asymmetry_correct.pvalue),
            "interpretation": (
                "Significant difference in the magnitude of interventions"
                if asymmetry_correct.pvalue < 0.05
                else "No significant difference in the magnitude of interventions"
            ),
        },
        "random_asymmetry": {
            "mean_pos": float(abs_pos_random.mean()),
            "mean_neg": float(abs_neg_random.mean()),
            "ratio": float(abs_pos_random.mean() / abs_neg_random.mean()),
            "t_stat": float(asymmetry_random.statistic),
            "p_value": float(asymmetry_random.pvalue),
            "interpretation": (
                "Significant difference in the magnitude of interventions"
                if asymmetry_random.pvalue < 0.05
                else "No significant difference in the magnitude of interventions"
            ),
        },
        "differential_asymmetry": {
            "t_stat": float(differential.statistic),
            "p_value": float(differential.pvalue),
            "interpretation": (
                "Significant difference in the magnitude of interventions"
                if differential.pvalue < 0.05
                else "No significant difference in the magnitude of interventions"
            ),
        },
    }
    
    
def separate_direction_ols(
    diff_pos: np.ndarray,
    diff_rand_pos: np.ndarray,
    diff_neg: np.ndarray,
    diff_rand_neg: np.ndarray,
    dataset: pd.DataFrame,
    additional_mask: np.ndarray | None = None,
) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    """Test each direction separately to capture asymmetric effects.
    
    Returns two models:
    - positive_model: Tests if positive intervention affects correct > random
    - negative_model: Tests if negative intervention affects correct > random
    """
    N = diff_pos.shape[0]
    r = dataset["real_object"].values[:N]
    c = dataset["correct"].values[:N]
    
    mask = (r == 1) & (c == 1)
    if additional_mask is not None:
        mask = mask & additional_mask
    M = mask.sum()
    
    # Model 1: Positive direction only
    df_pos = pd.DataFrame({
        "effect": np.concatenate([diff_pos[mask], diff_rand_pos[mask]]),
        "is_correct_token": [1] * M + [0] * M,
        "statement": np.tile(np.arange(M), 2),
    })
    
    y_pos, X_pos = patsy.dmatrices(
        "effect ~ is_correct_token",
        data=df_pos,
        return_type="dataframe",
    )
    groups_pos = df_pos.loc[X_pos.index, "statement"].to_numpy()
    model_pos = sm.OLS(y_pos, X_pos).fit(
        cov_type="cluster", cov_kwds={"groups": groups_pos}
    )
    
    # Model 2: Negative direction only
    df_neg = pd.DataFrame({
        "effect": np.concatenate([diff_neg[mask], diff_rand_neg[mask]]),
        "is_correct_token": [1] * M + [0] * M,
        "statement": np.tile(np.arange(M), 2),
    })
    
    y_neg, X_neg = patsy.dmatrices(
        "effect ~ is_correct_token",
        data=df_neg,
        return_type="dataframe",
    )
    groups_neg = df_neg.loc[X_neg.index, "statement"].to_numpy()
    model_neg = sm.OLS(y_neg, X_neg).fit(
        cov_type="cluster", cov_kwds={"groups": groups_neg}
    )
    
    return {
        "positive": model_pos,
        "negative": model_neg
    }

def check_ols_health(model: sm.regression.linear_model.RegressionResultsWrapper) -> dict:
    """Check numerical health of OLS model.
    
    OLS is closed-form (no iteration), but can have numerical issues:
    - Singular/near-singular design matrix
    - Extreme multicollinearity
    - NaN/Inf in estimates
    
    Returns:
        Dict with health indicators and overall is_healthy flag.
    """
    params = model.params.values
    bse = model.bse.values
    
    health = {
        # NaN/Inf checks
        "params_finite": bool(np.all(np.isfinite(params))),
        "stderr_finite": bool(np.all(np.isfinite(bse))),
        "stderr_positive": bool(np.all(bse > 0)),
        
        # Condition number (high = multicollinearity, >30 is concerning, >100 is bad)
        "condition_number": float(model.condition_number),
        "condition_ok": model.condition_number < 100,
        
        # Residual scale is valid
        "scale_finite": bool(np.isfinite(model.scale)),
        "scale_positive": bool(model.scale > 0),
        
        # R² in valid range
        "rsquared_valid": 0 <= model.rsquared <= 1,
    }
    
    # Overall health flag
    health["is_healthy"] = all([
        bool(health["params_finite"]),
        bool(health["stderr_finite"]),
        bool(health["stderr_positive"]),
        bool(health["condition_ok"]),
        bool(health["scale_finite"]),
        bool(health["scale_positive"]),
        bool(health["rsquared_valid"]),
    ])
    
    return health

##############################
####   Data Processors.   ####
##############################

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
        X: Hidden states to modify, shape [B, H] or [B, S, H].
        direction: Concept direction vector, shape [H].
        delta: Scalar translation magnitude (e.g. ±sigma).

    Returns:
        Translated embeddings with the same shape as X.

    """
    unit_dir = direction / direction.norm()
    return X + delta * unit_dir
