"""Tests for intervention utilities.

Tests cover all major functions in response/interventions_utils.py following
the AAA pattern (Arrange-Act-Assert) and isolating units with mocks where needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from response.interventions_utils import (
    InstructInterventionDataProcessor,
    InterventionDataProcessor,
    compute_layer_scale,
    diff_of_diff_ols,
    intervention_success_rate,
    mean_logprobs,
    normalize_logprob,
    random_answer_ids,
    sum_logprobs,
    translate_concept,
)


class TestRandomAnswerIds:
    """Test random answer ID generation."""

    def test_random_answer_ids_returns_correct_number_of_sequences(self):
        """Test that output has same length as input."""
        # Arrange
        seq_ids = [
            [torch.tensor([1, 2, 3])],
            [torch.tensor([4, 5])],
            [torch.tensor([6, 7, 8, 9])],
        ]
        vocab_size = 1000

        # Act
        result = random_answer_ids(seq_ids, vocab_size)

        # Assert
        assert len(result) == len(seq_ids)

    def test_random_answer_ids_returns_correct_lengths(self):
        """Test that each output sequence has correct length."""
        # Arrange
        seq_ids = [[torch.tensor([1, 2, 3])], [torch.tensor([4, 5])]]
        vocab_size = 1000

        # Act
        result = random_answer_ids(seq_ids, vocab_size)

        # Assert
        assert len(result[0]) == len(seq_ids[0])
        assert len(result[1]) == len(seq_ids[1])

    def test_random_answer_ids_values_within_vocab_size(self):
        """Test that generated IDs are within vocabulary bounds."""
        # Arrange
        seq_ids = [[torch.tensor([1, 2, 3])]]
        vocab_size = 100

        # Act
        result = random_answer_ids(seq_ids, vocab_size)

        # Assert
        assert torch.all(result[0] < vocab_size)
        assert torch.all(result[0] >= 0)

    def test_random_answer_ids_no_duplicates_within_sequence(self):
        """Test that IDs are sampled without replacement."""
        # Arrange
        seq_ids = [[torch.tensor([1, 2, 3, 4, 5])]]
        vocab_size = 100

        # Act
        result = random_answer_ids(seq_ids, vocab_size)

        # Assert
        unique_values = torch.unique(result[0])
        assert len(unique_values) == len(result[0])


class TestNormalizeLogprob:
    """Test log-probability normalization."""

    def test_normalize_logprob_scalar_returns_same(self):
        """Test that scalar input is returned unchanged."""
        # Arrange
        lp = torch.tensor(-2.5)

        # Act
        result = normalize_logprob(lp)

        # Assert
        assert result == lp

    def test_normalize_logprob_vector_returns_sum(self):
        """Test that vector input is summed."""
        # Arrange
        lp = torch.tensor([-1.0, -2.0, -3.0])

        # Act
        result = normalize_logprob(lp)

        # Assert
        assert result == torch.tensor(-6.0)

    def test_normalize_logprob_empty_tensor(self):
        """Test behavior with empty tensor."""
        # Arrange
        lp = torch.tensor([])

        # Act
        result = normalize_logprob(lp)

        # Assert
        assert result == torch.tensor(0.0)


class TestMeanLogprobs:
    """Test mean log-probability computation."""

    def test_mean_logprobs_with_scalars(self):
        """Test averaging of scalar log-probabilities."""
        # Arrange
        logprobs = [torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-3.0)]

        # Act
        result = mean_logprobs(logprobs)

        # Assert
        assert result == pytest.approx(-2.0)

    def test_mean_logprobs_with_vectors(self):
        """Test averaging with multi-token sequences."""
        # Arrange
        logprobs = [
            torch.tensor([-1.0, -1.0]),  # sum = -2.0
            torch.tensor([-2.0, -2.0]),  # sum = -4.0
        ]

        # Act
        result = mean_logprobs(logprobs)

        # Assert
        assert result == pytest.approx(-3.0)

    def test_mean_logprobs_single_value(self):
        """Test with single log-probability."""
        # Arrange
        logprobs = [torch.tensor(-2.5)]

        # Act
        result = mean_logprobs(logprobs)

        # Assert
        assert result == pytest.approx(-2.5)


class TestSumLogprobs:
    """Test total log-likelihood computation."""

    def test_sum_logprobs_with_scalars(self):
        """Test summing scalar log-probabilities."""
        # Arrange
        logprobs = [torch.tensor(-1.0), torch.tensor(-2.0), torch.tensor(-3.0)]

        # Act
        result = sum_logprobs(logprobs)

        # Assert
        assert result == pytest.approx(-6.0)

    def test_sum_logprobs_with_vectors(self):
        """Test summing with multi-token sequences."""
        # Arrange
        logprobs = [
            torch.tensor([-1.0, -1.0]),  # sum = -2.0
            torch.tensor([-2.0, -2.0]),  # sum = -4.0
        ]

        # Act
        result = sum_logprobs(logprobs)

        # Assert
        assert result == pytest.approx(-6.0)


class TestComputeLayerScale:
    """Test layer scale computation."""

    def test_compute_layer_scale_returns_positive_value(self):
        """Test that scale is positive."""
        # Arrange
        mock_dh = Mock()
        X = torch.randn(10, 512)
        mock_dh.train_bags.return_value = {"last_embedding": X}
        direction = torch.randn(512)
        layer_id = 5

        # Act
        result = compute_layer_scale(mock_dh, direction, layer_id)

        # Assert
        assert result > 0

    def test_compute_layer_scale_with_eps_clamping(self):
        """Test that result is clamped to minimum epsilon."""
        # Arrange
        mock_dh = Mock()
        X = torch.zeros(10, 512)  # All zeros will give zero std
        mock_dh.train_bags.return_value = {"last_embedding": X}
        direction = torch.randn(512)
        layer_id = 5
        eps = 1e-3

        # Act
        result = compute_layer_scale(mock_dh, direction, layer_id, eps=eps)

        # Assert
        assert result >= eps

    def test_compute_layer_scale_raises_on_wrong_shape(self):
        """Test that ValueError is raised for incorrect tensor shape."""
        # Arrange
        mock_dh = Mock()
        X = torch.randn(10, 5, 512)  # 3D instead of 2D
        mock_dh.train_bags.return_value = {"last_embedding": X}
        direction = torch.randn(512)
        layer_id = 5

        # Act & Assert
        with pytest.raises(ValueError, match="Expected X to be"):
            compute_layer_scale(mock_dh, direction, layer_id)


class TestDiffOfDiffOls:
    """Test difference-in-differences OLS analysis."""

    def test_diff_of_diff_ols_returns_model(self):
        """Test that function returns fitted OLS model."""
        # Arrange
        N = 20
        diff_pos = np.random.randn(N)
        diff_neg = np.random.randn(N)
        diff_rand_pos = np.random.randn(N)
        diff_rand_neg = np.random.randn(N)
        dataset = pd.DataFrame(
            {"real_object": np.ones(N), "correct": np.ones(N), "statement_id": range(N)}
        )

        # Act
        result = diff_of_diff_ols(
            diff_pos, diff_neg, diff_rand_pos, diff_rand_neg, dataset
        )

        # Assert
        assert hasattr(result, "params")
        assert hasattr(result, "pvalues")
        assert hasattr(result, "conf_int")

    def test_diff_of_diff_ols_has_interaction_term(self):
        """Test that model includes the key interaction coefficient."""
        # Arrange
        N = 30
        diff_pos = np.random.randn(N)
        diff_neg = np.random.randn(N)
        diff_rand_pos = np.random.randn(N)
        diff_rand_neg = np.random.randn(N)
        dataset = pd.DataFrame(
            {"real_object": np.ones(N), "correct": np.ones(N), "statement_id": range(N)}
        )

        # Act
        result = diff_of_diff_ols(
            diff_pos, diff_neg, diff_rand_pos, diff_rand_neg, dataset
        )

        # Assert
        assert "is_correct_token:is_pos_translation" in result.params.index

    def test_diff_of_diff_ols_filters_dataset(self):
        """Test that only real, correct statements are used."""
        # Arrange
        N = 40
        diff_pos = np.random.randn(N)
        diff_neg = np.random.randn(N)
        diff_rand_pos = np.random.randn(N)
        diff_rand_neg = np.random.randn(N)
        dataset = pd.DataFrame(
            {
                "real_object": [1] * 20 + [0] * 20,
                "correct": [1] * 10 + [0] * 10 + [1] * 20,
                "statement_id": range(N),
            }
        )

        # Act
        result = diff_of_diff_ols(
            diff_pos, diff_neg, diff_rand_pos, diff_rand_neg, dataset
        )

        # Assert - only first 10 should be included (4 obs per statement)
        assert result.nobs == 40  # 10 statements × 4 conditions


class TestInterventionSuccessRate:
    """Test intervention success rate calculation."""

    def test_intervention_success_rate_all_successful(self):
        """Test with all statements showing successful intervention."""
        # Arrange - positive and negative always opposite, positive aligns with dominant
        diff_pos = np.array([1.0, 1.0, 1.0, 1.0])
        diff_neg = np.array([-1.0, -1.0, -1.0, -1.0])

        # Act
        result = intervention_success_rate(diff_pos, diff_neg)

        # Assert
        assert result["success_rate"] == 1.0
        assert result["n_success"] == 4
        assert result["opposition_rate"] == 1.0

    def test_intervention_success_rate_no_success(self):
        """Test with no successful interventions."""
        # Arrange - same direction for both interventions
        diff_pos = np.array([1.0, 1.0, 1.0])
        diff_neg = np.array([1.0, 1.0, 1.0])

        # Act
        result = intervention_success_rate(diff_pos, diff_neg)

        # Assert
        assert result["success_rate"] == 0.0
        assert result["opposition_rate"] == 0.0

    def test_intervention_success_rate_with_filtering(self):
        """Test that dataset filtering works correctly."""
        # Arrange
        diff_pos = np.array([1.0, 1.0, -1.0, 1.0])
        diff_neg = np.array([-1.0, -1.0, 1.0, -1.0])
        dataset = pd.DataFrame(
            {
                "real_object": [1, 1, 0, 1],
                "correct": [1, 0, 1, 1],
            }
        )

        # Act
        result = intervention_success_rate(diff_pos, diff_neg, dataset=dataset)

        # Assert - only rows 0 and 3 should be included
        assert result["n_total"] == 2

    def test_intervention_success_rate_eps_threshold(self):
        """Test that small values are treated as zero."""
        # Arrange
        diff_pos = np.array([1e-15, 1.0])
        diff_neg = np.array([-1e-15, -1.0])
        eps = 1e-12

        # Act
        result = intervention_success_rate(diff_pos, diff_neg, eps=eps)

        # Assert - first statement has zero effects, only second is valid
        assert result["zero_effect_rate"] > 0

    def test_intervention_success_rate_p_value(self):
        """Test that p-value is computed."""
        # Arrange
        diff_pos = np.array([1.0] * 10 + [-1.0] * 10)
        diff_neg = np.array([-1.0] * 10 + [1.0] * 10)

        # Act
        result = intervention_success_rate(diff_pos, diff_neg)

        # Assert
        assert "p_value" in result
        assert 0 <= result["p_value"] <= 1

    def test_intervention_success_rate_dominant_direction(self):
        """Test dominant direction identification."""
        # Arrange - more positive effects
        diff_pos = np.array([1.0, 1.0, 1.0, -1.0])
        diff_neg = np.array([-1.0, -1.0, -1.0, 1.0])

        # Act
        result = intervention_success_rate(diff_pos, diff_neg)

        # Assert
        assert result["dominant_direction"] == 1.0


class TestTranslateConcept:
    """Test concept translation function."""

    def test_translate_concept_shape_preservation(self):
        """Test that output has same shape as input."""
        # Arrange
        X = torch.randn(2, 3, 512)
        direction = torch.randn(512)
        delta = 1.5

        # Act
        result = translate_concept(X, direction, delta)

        # Assert
        assert result.shape == X.shape

    def test_translate_concept_positive_delta(self):
        """Test translation with positive delta."""
        # Arrange
        X = torch.zeros(1, 1, 10)
        direction = torch.ones(10)
        delta = 2.0

        # Act
        result = translate_concept(X, direction, delta)

        # Assert - should add normalized direction * delta
        expected_shift = delta / direction.norm()
        assert torch.allclose(result[0, 0, 0], expected_shift, atol=1e-5)

    def test_translate_concept_negative_delta(self):
        """Test translation with negative delta."""
        # Arrange
        X = torch.zeros(1, 1, 10)
        direction = torch.ones(10)
        delta = -2.0

        # Act
        result = translate_concept(X, direction, delta)

        # Assert - should subtract
        expected_shift = delta / direction.norm()
        assert torch.allclose(result[0, 0, 0], expected_shift, atol=1e-5)

    def test_translate_concept_zero_delta(self):
        """Test that zero delta leaves input unchanged."""
        # Arrange
        X = torch.randn(2, 3, 512)
        direction = torch.randn(512)
        delta = 0.0

        # Act
        result = translate_concept(X, direction, delta)

        # Assert
        assert torch.allclose(result, X)

    def test_translate_concept_normalizes_direction(self):
        """Test that direction is normalized before application."""
        # Arrange
        X = torch.zeros(1, 1, 10)
        direction = torch.ones(10) * 5.0  # Non-unit norm
        delta = 1.0

        # Act
        result = translate_concept(X, direction, delta)

        # Assert - effect should be same as with unit direction
        unit_dir = direction / direction.norm()
        expected = X + delta * unit_dir.view(1, 1, -1)
        assert torch.allclose(result, expected)


class TestInterventionDataProcessor:
    """Test InterventionDataProcessor class."""

    def test_template_city_locations(self):
        """Test city locations template formatting."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(mock_dh, mock_tokenizer, "city_locations")

        # Act
        result = processor.template("Paris", "France", negation=0)

        # Assert
        assert "Paris" in result
        assert "located in" in result

    def test_template_med_indications(self):
        """Test medical indications template formatting."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(
            mock_dh, mock_tokenizer, "med_indications"
        )

        # Act
        result = processor.template("aspirin", "pain", negation=0)

        # Assert
        assert "Aspirin" in result
        assert "indicated for the treatment of" in result

    def test_template_with_negation(self):
        """Test template with negation flag."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(mock_dh, mock_tokenizer, "city_locations")

        # Act
        result = processor.template("Paris", "Germany", negation=1)

        # Assert
        assert "is not" in result

    def test_template_word_definitions(self):
        """Test word definitions template with categories."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(
            mock_dh, mock_tokenizer, "word_definitions"
        )

        # Act
        result_instance = processor.template(
            "dog", "animal", negation=0, category="instances"
        )
        result_synonym = processor.template(
            "happy", "joyful", negation=0, category="synonyms"
        )
        result_type = processor.template(
            "poodle", "dog", negation=0, category="types"
        )

        # Assert
        assert "dog is a" == result_instance
        assert "synonym" in result_synonym
        assert "type of" in result_type

    def test_template_invalid_datapack(self):
        """Test that invalid datapack raises ValueError."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(
            mock_dh, mock_tokenizer, "invalid_datapack"
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid data pack"):
            processor.template("obj1", "obj2", negation=0)

    def test_return_processed_test_df(self):
        """Test test dataframe processing."""
        # Arrange
        mock_dh = Mock()
        test_df = pd.DataFrame(
            {
                "object_1": ["Paris", "London"],
                "object_2": ["France", "UK"],
                "correct_object_2": ["France", "UK"],
                "real_object": [1, 1],
                "correct": [1, 1],
                "negation": [0, 0],
                "category": [None, None],
            }
        )
        mock_dh.get_test_df.return_value = test_df
        mock_tokenizer = Mock()
        processor = InterventionDataProcessor(mock_dh, mock_tokenizer, "city_locations")

        # Act
        result = processor.return_processed_test_df()

        # Assert
        assert "statement" in result.columns
        assert "answer" in result.columns
        assert len(result) == 2

    def test_get_answer_ids(self):
        """Test answer tokenization."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = Mock(input_ids=torch.tensor([[1, 2, 3]]))
        processor = InterventionDataProcessor(mock_dh, mock_tokenizer, "city_locations")

        # Act
        result = processor.get_answer_ids("France")

        # Assert
        mock_tokenizer.assert_called_once()
        assert torch.is_tensor(result)

    def test_get_answer_seq_ids(self):
        """Test incremental sequence generation for multi-token answers."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.tokenize.return_value = ["Paris", "is", "located", "in"]
        mock_tokenizer.convert_tokens_to_ids.return_value = [1, 2, 3, 4]
        processor = InterventionDataProcessor(mock_dh, mock_tokenizer, "city_locations")

        # Act
        statements, answers, answer_ids, init_ids = processor.get_answer_seq_ids(
            "Paris is located in", "France and Europe"
        )

        # Assert
        assert len(statements) > 0
        assert len(answers) > 0
        assert len(answer_ids) > 0


class TestInstructInterventionDataProcessor:
    """Test InstructInterventionDataProcessor class."""

    def test_instruct_template_with_system_role(self):
        """Test instruction template formatting with separate system role."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InstructInterventionDataProcessor(
            mock_dh,
            mock_tokenizer,
            "city_locations",
            user_role="user",
            system_role="system",
            assist_role="assistant",
        )

        # Act
        result = processor._instruct_template("Paris is located in")

        # Assert
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "Paris is located in" in result[1]["content"]

    def test_instruct_template_same_user_system(self):
        """Test instruction template when system and user roles are same."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        processor = InstructInterventionDataProcessor(
            mock_dh,
            mock_tokenizer,
            "city_locations",
            user_role="user",
            system_role="user",
            assist_role="assistant",
        )

        # Act
        result = processor._instruct_template("Paris is located in")

        # Assert
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_return_processed_test_df_instruct(self):
        """Test that statements are formatted as instruction messages."""
        # Arrange
        mock_dh = Mock()
        test_df = pd.DataFrame(
            {
                "object_1": ["Paris"],
                "object_2": ["France"],
                "correct_object_2": ["France"],
                "real_object": [1],
                "correct": [1],
                "negation": [0],
                "category": [None],
            }
        )
        mock_dh.get_test_df.return_value = test_df
        mock_tokenizer = Mock()
        processor = InstructInterventionDataProcessor(
            mock_dh,
            mock_tokenizer,
            "city_locations",
            user_role="user",
            system_role="system",
            assist_role="assistant",
        )

        # Act
        result = processor.return_processed_test_df()

        # Assert
        assert isinstance(result.iloc[0]["statement"], list)
        assert all("role" in msg for msg in result.iloc[0]["statement"])

    def test_get_answer_seq_ids_instruct(self):
        """Test incremental sequences with instruction format."""
        # Arrange
        mock_dh = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.side_effect = [
            "templated statement",
            [1, 2, 3, 4],
        ]
        mock_tokenizer.convert_tokens_to_ids.return_value = [5, 6]
        mock_tokenizer.tokenize.return_value = ["France"]

        processor = InstructInterventionDataProcessor(
            mock_dh,
            mock_tokenizer,
            "city_locations",
            user_role="user",
            system_role="system",
            assist_role="assistant",
        )

        statement_msg = [{"role": "user", "content": "Paris is located in"}]

        # Act
        statements, answers, answer_ids, init_ids = processor.get_answer_seq_ids(
            statement_msg, "France"
        )

        # Assert
        assert len(statements) > 0
        assert len(answers) > 0
        mock_tokenizer.apply_chat_template.assert_called()
