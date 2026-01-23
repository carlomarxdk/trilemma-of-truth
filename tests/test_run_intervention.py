"""Tests for run_intervention.py.

Tests cover configuration validation, checkpointing, and result saving
following the AAA pattern and using mocks to isolate units.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import torch
from omegaconf import DictConfig, OmegaConf

from run_intervention import checkpointing, save, validate_config


class TestValidateConfig:
    """Test configuration validation."""

    def test_validate_config_with_valid_list_datasets(self):
        """Test validation passes with valid list of datasets."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc", "med_indications"]},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
                "output_dir": "/tmp/output",
                "probe_dir": "/tmp/probes",
            }
        )

        # Act & Assert - should not raise
        validate_config(cfg)

    def test_validate_config_raises_on_non_list_datasets(self):
        """Test that non-list datasets raise ValueError."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": "cities_loc"},  # String instead of list
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
            }
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Datasets must be a list"):
            validate_config(cfg)

    def test_validate_config_raises_on_empty_datasets(self):
        """Test that empty datasets list raises ValueError."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": []},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
            }
        )

        # Act & Assert
        with pytest.raises(ValueError, match="At least one dataset"):
            validate_config(cfg)

    def test_validate_config_raises_on_invalid_layer_range(self):
        """Test that invalid layer range raises ValueError."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10, 20],  # Three values instead of two
                "output_dir": "/tmp/output",
                "probe_dir": "/tmp/probes",
            }
        )

        # Act & Assert
        with pytest.raises(ValueError, match="Layer range must be a list of two"):
            validate_config(cfg)

    def test_validate_config_modifies_trial_name_with_search(self):
        """Test that trial name is modified when search=True."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 3,
                "search": True,
                "layer_range": [0, 10],
                "output_dir": "/tmp/output",
                "probe_dir": "/tmp/probes",
            }
        )

        # Act
        validate_config(cfg)

        # Assert
        assert "_search" in cfg.trial_name
        assert "_task-3" in cfg.trial_name

    def test_validate_config_modifies_trial_name_without_search(self):
        """Test that trial name includes task but not search when search=False."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 2,
                "search": False,
                "layer_range": [0, 10],
                "output_dir": "/tmp/output",
                "probe_dir": "/tmp/probes",
            }
        )

        # Act
        validate_config(cfg)

        # Assert
        assert "_search" not in cfg.trial_name
        assert "_task-2" in cfg.trial_name

    def test_validate_config_sets_device_when_none(self):
        """Test that device is set when initially None."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": None,
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
                "output_dir": "/tmp/output",
                "probe_dir": "/tmp/probes",
            }
        )

        # Act
        with patch("run_intervention.get_device", return_value="cuda:0"):
            validate_config(cfg)

        # Assert
        assert cfg.device is not None

    def test_validate_config_updates_output_and_probe_dir(self):
        """Test that output_dir and probe_dir are updated with trial_name."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": "cpu",
                "trial_name": "test_trial",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
                "output_dir": "/base/output",
                "probe_dir": "/base/probes",
            }
        )

        # Act
        validate_config(cfg)

        # Assert
        assert cfg.trial_name in cfg.output_dir
        assert cfg.trial_name in cfg.probe_dir


class TestCheckpointing:
    """Test checkpointing functionality."""

    def test_checkpointing_returns_all_layers_when_none_completed(self):
        """Test that all layers are returned when starting fresh."""
        # Arrange
        cfg = DictConfig({"output_dir": "/tmp/test_output_empty"})
        existing_layers = [0, 5, 10, 15, 20]

        # Act
        with patch("run_intervention.glob", return_value=[]):
            missing_layers = checkpointing(cfg, existing_layers)

        # Assert
        assert missing_layers == existing_layers

    def test_checkpointing_returns_missing_layers(self):
        """Test that only incomplete layers are returned."""
        # Arrange
        cfg = DictConfig({"output_dir": "/tmp/test_output"})
        existing_layers = [0, 5, 10, 15, 20]
        completed_files = [
            "/tmp/test_output/layer_0",
            "/tmp/test_output/layer_10",
        ]

        # Act
        with patch("run_intervention.glob", return_value=completed_files):
            missing_layers = checkpointing(cfg, existing_layers)

        # Assert
        assert set(missing_layers) == {5, 15, 20}
        assert missing_layers == sorted(missing_layers)

    def test_checkpointing_returns_empty_when_all_completed(self):
        """Test that empty list is returned when all layers complete."""
        # Arrange
        cfg = DictConfig({"output_dir": "/tmp/test_output"})
        existing_layers = [0, 5, 10]
        completed_files = [
            "/tmp/test_output/layer_0",
            "/tmp/test_output/layer_5",
            "/tmp/test_output/layer_10",
        ]

        # Act
        with patch("run_intervention.glob", return_value=completed_files):
            missing_layers = checkpointing(cfg, existing_layers)

        # Assert
        assert missing_layers == []

    def test_checkpointing_handles_non_numeric_files(self):
        """Test that non-matching files are ignored."""
        # Arrange
        cfg = DictConfig({"output_dir": "/tmp/test_output"})
        existing_layers = [0, 5, 10]
        completed_files = [
            "/tmp/test_output/layer_5",
            "/tmp/test_output/other_file.txt",
            "/tmp/test_output/layer_metadata.json",
        ]

        # Act
        with patch("run_intervention.glob", return_value=completed_files):
            missing_layers = checkpointing(cfg, existing_layers)

        # Assert
        assert 5 not in missing_layers
        assert set(missing_layers) == {0, 10}

    def test_checkpointing_sorted_output(self):
        """Test that output is sorted."""
        # Arrange
        cfg = DictConfig({"output_dir": "/tmp/test_output"})
        existing_layers = [20, 5, 15, 0, 10]
        completed_files = ["/tmp/test_output/layer_5"]

        # Act
        with patch("run_intervention.glob", return_value=completed_files):
            missing_layers = checkpointing(cfg, existing_layers)

        # Assert
        assert missing_layers == sorted(missing_layers)
        assert missing_layers == [0, 10, 15, 20]


class TestSave:
    """Test save function."""

    def test_save_creates_output_dict_with_all_keys(self):
        """Test that output dictionary contains all expected keys."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_save", "save_results": False, "task": 0}
        )
        layer_id = 10
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.75, "p_value": 0.01}
        arrays = [np.array([1.0, 2.0, 3.0])] * 6

        # Act
        with patch("run_intervention.log"):
            save(cfg, layer_id, did_result, success_results, *arrays)

        # Should not raise and log should be called

    def test_save_creates_directories_when_save_results_true(self):
        """Test that directories are created when saving."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_save_dir", "save_results": True, "task": 0}
        )
        layer_id = 5
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.8, "p_value": 0.05}
        arrays = [np.random.randn(10)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ) as mock_mkdir, patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ):
            save(cfg, layer_id, did_result, success_results, *arrays)

            # Assert
            mock_mkdir.assert_called()

    def test_save_writes_numpy_arrays_when_save_results_true(self):
        """Test that numpy arrays are saved to disk."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_save_arrays", "save_results": True, "task": 0}
        )
        layer_id = 12
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.9, "p_value": 0.001}
        s_orig = np.array([1.0, 2.0])
        s_neg = np.array([3.0, 4.0])
        s_pos = np.array([5.0, 6.0])
        r_orig = np.array([7.0, 8.0])
        r_neg = np.array([9.0, 10.0])
        r_pos = np.array([11.0, 12.0])

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save") as mock_save, patch(
            "run_intervention._atomic_write_json"
        ):
            save(
                cfg, layer_id, did_result, success_results, s_orig, s_neg, s_pos,
                r_orig, r_neg, r_pos
            )

            # Assert - should save 6 arrays
            assert mock_save.call_count == 6

    def test_save_writes_json_summary_when_save_results_true(self):
        """Test that JSON summary is written."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_save_json", "save_results": True, "task": 0}
        )
        layer_id = 8
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.65, "p_value": 0.1}
        arrays = [np.random.randn(10)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ) as mock_write_json:
            save(cfg, layer_id, did_result, success_results, *arrays)

            # Assert
            mock_write_json.assert_called_once()
            call_args = mock_write_json.call_args
            output_dict = call_args[0][1]
            assert "did" in output_dict
            assert "descriptives" in output_dict
            assert "success_results" in output_dict

    def test_save_does_not_write_when_save_results_false(self):
        """Test that files are not written when save_results=False."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_no_save", "save_results": False, "task": 0}
        )
        layer_id = 3
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.5, "p_value": 0.5}
        arrays = [np.random.randn(10)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.np.save"
        ) as mock_save, patch("run_intervention._atomic_write_json") as mock_write_json:
            save(cfg, layer_id, did_result, success_results, *arrays)

            # Assert
            mock_save.assert_not_called()
            mock_write_json.assert_not_called()

    def test_save_includes_interaction_coefficient(self):
        """Test that DiD interaction term is included in output."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_interaction", "save_results": True, "task": 0}
        )
        layer_id = 7
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.7, "p_value": 0.02}
        arrays = [np.random.randn(10)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ) as mock_write_json:
            save(cfg, layer_id, did_result, success_results, *arrays)

            output_dict = mock_write_json.call_args[0][1]

            # Assert
            assert "interaction_coef" in output_dict["did"]
            assert "interaction_pval" in output_dict["did"]
            assert "interaction_ci" in output_dict["did"]

    def test_save_includes_descriptive_statistics(self):
        """Test that descriptive statistics are included."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_descriptives", "save_results": True, "task": 0}
        )
        layer_id = 9
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.85, "p_value": 0.005}
        s_orig = np.array([1.0, 2.0, 3.0])
        s_neg = np.array([1.5, 2.5, 3.5])
        s_pos = np.array([0.5, 1.5, 2.5])
        r_orig = np.array([0.1, 0.2, 0.3])
        r_neg = np.array([0.15, 0.25, 0.35])
        r_pos = np.array([0.05, 0.15, 0.25])

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ) as mock_write_json:
            save(
                cfg, layer_id, did_result, success_results, s_orig, s_neg, s_pos,
                r_orig, r_neg, r_pos
            )

            output_dict = mock_write_json.call_args[0][1]

            # Assert
            desc = output_dict["descriptives"]
            assert "correct_orig_mean" in desc
            assert "correct_pos_mean" in desc
            assert "correct_neg_mean" in desc
            assert "random_orig_mean" in desc
            assert "random_pos_mean" in desc
            assert "random_neg_mean" in desc
            assert desc["correct_orig_mean"] == pytest.approx(2.0)

    def test_save_includes_significance_flag(self):
        """Test that significance flag is computed correctly."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_signf", "save_results": True, "task": 0}
        )
        layer_id = 11
        did_result = self._create_mock_did_result(interaction_pval=0.03)
        success_results = {"success_rate": 0.95, "p_value": 0.001}
        arrays = [np.random.randn(10)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ) as mock_write_json:
            save(cfg, layer_id, did_result, success_results, *arrays)

            output_dict = mock_write_json.call_args[0][1]

            # Assert - p=0.03 < 0.05, so should be significant
            assert output_dict["did"]["signf"] == 1

    def test_save_includes_model_stats(self):
        """Test that model statistics are included."""
        # Arrange
        cfg = DictConfig(
            {"output_dir": "/tmp/test_model_stats", "save_results": True, "task": 0}
        )
        layer_id = 14
        did_result = self._create_mock_did_result()
        success_results = {"success_rate": 0.6, "p_value": 0.08}
        arrays = [np.random.randn(20)] * 6

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir"
        ), patch("run_intervention.np.save"), patch(
            "run_intervention._atomic_write_json"
        ) as mock_write_json:
            save(cfg, layer_id, did_result, success_results, *arrays)

            output_dict = mock_write_json.call_args[0][1]

            # Assert
            assert "r_squared" in output_dict["did"]
            assert "df_resid" in output_dict["did"]
            assert "n_obs" in output_dict["did"]
            assert "n_statements" in output_dict["did"]

    @staticmethod
    def _create_mock_did_result(interaction_pval=0.01):
        """Create a mock DiD OLS result object."""
        mock_result = Mock()
        
        # Parameters
        params_dict = {
            "Intercept": 0.5,
            "is_correct_token": 0.3,
            "is_pos_translation": 0.2,
            "is_correct_token:is_pos_translation": 0.4,
        }
        mock_result.params = pd.Series(params_dict)
        
        # Standard errors
        bse_dict = {
            "Intercept": 0.05,
            "is_correct_token": 0.03,
            "is_pos_translation": 0.02,
            "is_correct_token:is_pos_translation": 0.04,
        }
        mock_result.bse = pd.Series(bse_dict)
        
        # P-values
        pvalues_dict = {
            "Intercept": 0.001,
            "is_correct_token": 0.001,
            "is_pos_translation": 0.001,
            "is_correct_token:is_pos_translation": interaction_pval,
        }
        mock_result.pvalues = pd.Series(pvalues_dict)
        
        # T-values
        tvalues_dict = {
            "Intercept": 10.0,
            "is_correct_token": 10.0,
            "is_pos_translation": 10.0,
            "is_correct_token:is_pos_translation": 10.0,
        }
        mock_result.tvalues = pd.Series(tvalues_dict)
        
        # Confidence intervals
        ci_df = pd.DataFrame(
            {
                0: [0.4, 0.24, 0.16, 0.32],
                1: [0.6, 0.36, 0.24, 0.48],
            },
            index=[
                "Intercept",
                "is_correct_token",
                "is_pos_translation",
                "is_correct_token:is_pos_translation",
            ],
        )
        mock_result.conf_int.return_value = ci_df
        
        # Model stats
        mock_result.rsquared = 0.45
        mock_result.df_resid = 100
        mock_result.nobs = 120
        
        return mock_result


class TestIntegration:
    """Integration tests for main workflow components."""

    def test_validate_and_checkpointing_workflow(self):
        """Test that validate and checkpointing work together."""
        # Arrange
        cfg = OmegaConf.create(
            {
                "datapack": {"datasets": ["cities_loc"]},
                "device": "cpu",
                "trial_name": "integration_test",
                "task": 0,
                "search": False,
                "layer_range": [0, 10],
                "output_dir": "/tmp/integration",
                "probe_dir": "/tmp/probes",
            }
        )
        existing_layers = [0, 5, 10, 15, 20]
        completed_files = ["/tmp/integration_test_task-0/layer_5"]

        # Act
        validate_config(cfg)
        with patch("run_intervention.glob", return_value=completed_files):
            missing = checkpointing(cfg, existing_layers)

        # Assert
        assert "task-0" in cfg.trial_name
        assert 5 not in missing
        assert set(missing) == {0, 10, 15, 20}

    def test_save_creates_proper_file_structure(self):
        """Test that save creates the expected file structure."""
        # Arrange
        cfg = DictConfig(
            {
                "output_dir": "/tmp/test_structure",
                "save_results": True,
                "task": 2,
            }
        )
        layer_id = 8
        did_result = TestSave._create_mock_did_result()
        success_results = {"success_rate": 0.75, "p_value": 0.01}
        arrays = [np.random.randn(15)] * 6

        created_paths = []

        def track_mkdir(self, *args, **kwargs):
            created_paths.append(str(self))

        def track_save(path, array):
            created_paths.append(str(path))

        def track_json(path, data):
            created_paths.append(str(path))

        # Act
        with patch("run_intervention.log"), patch(
            "run_intervention.Path.mkdir", track_mkdir
        ), patch("run_intervention.np.save", track_save), patch(
            "run_intervention._atomic_write_json", track_json
        ):
            save(cfg, layer_id, did_result, success_results, *arrays)

        # Assert - check that we created score directory and saved arrays
        score_paths = [p for p in created_paths if "scores" in p]
        array_paths = [p for p in created_paths if ".npy" in p]
        json_paths = [p for p in created_paths if ".json" in p]

        assert len(score_paths) > 0
        assert len(array_paths) == 6  # 6 arrays saved
        assert len(json_paths) == 1  # 1 JSON summary
