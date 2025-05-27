import polars as pl
from typing import List
import torch
import numpy as np
from glob import glob
import logging
from dataclasses import dataclass, field
LEGAL_ACTIVATION_TYPES = ["last", "full"]

log = logging.getLogger(__name__)


def shape_as_tuple(x):
    d = x.shape[0]
    if d == 3:
        return (x[0], x[1], x[2])
    elif d == 2:
        return (x[0], x[1])
    else:
        raise Exception("Number of dimensions is too low")


def remove_padded(x):
    # padded rows are always first
    mask = (x - x[0]).sum(1) != 0
    return x[mask]


def stack_tensors(tensors, padding_value=0, max_length=None):
    """
    Stack 3D tensors with different lengths along the 1st dimension, padding with a specified value when needed.

    Args:
        tensors (list of torch.Tensor): List of 3D tensors to be stacked.
        padding_value (float, optional): Value to use for padding. Defaults to 0.
        max_length (int, optional): The length to pad all tensors to. If None, it will use the maximum length found in the tensors.

    Returns:
        torch.Tensor: A single stacked tensor with padding applied along the 1st dimension.
    """
    # Determine the maximum length of the 1st dimension across all tensors
    if max_length is None:
        max_length = max(tensor.size(1) for tensor in tensors)

    # Pad each tensor along the 1st dimension to the maximum length
    padded_tensors = []
    for tensor in tensors:
        pad_size = max_length - tensor.size(1)
        if pad_size > 0:
            # Create a padding tensor with the required size
            pad_tensor = torch.full(
                (tensor.size(0), pad_size, tensor.size(2)), padding_value, dtype=tensor.dtype)
            padded_tensor = torch.cat((tensor, pad_tensor), dim=1)
        else:
            padded_tensor = tensor
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors along the 0th dimension
    stacked_tensor = torch.vstack(padded_tensors)
    return stacked_tensor


@dataclass
class DataHandler:
    model: str = field(default="llama-3-8b",
                       metadata={"help": "The LLM use in the project."})
    datasets: List[str] = field(default_factory=lambda: ["cities_combined", "cities_fake"],
                                metadata={
                                "help": "The datasets to be used in the project."})
    activation_type: str = field(default="last"),
    dataset_path: str = field(default="datasets/")
    activations_path: str = field(default="outputs/activations/")
    with_calibration: bool = field(default=False,
                                   metadata={"help": "Whether to include a calibration set."})
    load_scores: str = field(default=''),
    output_path: str = field(default="outputs/")
    verbose: bool = field(default=True)

    def __post_init__(self):
        assert self.activation_type in LEGAL_ACTIVATION_TYPES, f"Activation type must be either {LEGAL_ACTIVATION_TYPES}."

    def assemble(self, exclusive_split: bool, test_size: float = 0.2, calibration_size: float = 0.2, seed: int = 42, shuffle: bool = True):
        """
        Assemble the data for the project.
        """
        data = []
        columns = self.column_list()
        for dataset in self.datasets:
            # Load the dataset
            df = pl.read_csv(f"{self.dataset_path}{dataset}.csv")
            df = df.with_columns([
                pl.col("correct").cast(pl.Int32()),
                pl.col("negation").cast(pl.Int32()),
                pl.col("real_object").cast(pl.Int32()),
            ])
            if self.load_scores != '':
                try:
                    scores = np.load(
                        f"outputs/probes/prompt/{self.load_scores}/{self.model}/{dataset}/scores.npy")
                    n_scores = scores.shape[-1]
                    df = df.with_columns(
                        [pl.Series(f"scores_{i}", scores[:, i]) for i in range(n_scores)])
                except Exception as e:
                    log.error(e)
                    log.error(
                        f"Scores not found for {self.model} and {dataset}.")
            missing_columns = set(columns).difference(set(df.columns))
            for column in missing_columns:
                df = df.with_columns(pl.lit(0.0).alias(column))
            data.append(df.select(sorted(df.columns))
                        )

        # Concatenate the datasets
        data = pl.concat(data, how="vertical_relaxed").to_pandas()

        self.data = data
        if self.verbose:
            log.warning("Datasets assembled.")
        if exclusive_split:
            self.__exclusive_data_split__(
                test_size=test_size, calibration_size=calibration_size, seed=seed, shuffle=shuffle)
        else:
            self.__data_split__(
                test_size=test_size, calibration_size=calibration_size, seed=seed, shuffle=shuffle)

    def __exclusive_data_split__(self, test_size: float = 0.2, calibration_size: float = 0.2, seed: int = 42, shuffle: bool = True):
        """
        Split the data into training and testing sets: makes sure that objects from train set do not appear in test set.
        """
        df_train, df_test = self.__generate_exclusive_split__(
            self.data, test_size, seed)

        if self.with_calibration:
            df_train, df_calib = self.__generate_exclusive_split__(
                df_train, calibration_size, seed)
            self.calibration_ids = np.array(df_calib.index)
        else:
            self.calibration_ids = None

        self.train_ids = np.array(df_train.index)
        self.test_ids = np.array(df_test.index)
        if shuffle:
            np.random.seed(seed + 1)
            np.random.shuffle(self.train_ids)
            np.random.shuffle(self.test_ids)
            if self.with_calibration:
                np.random.shuffle(self.calibration_ids)

        if self.verbose:
            train_size_ratio = len(self.train_ids)/self.data.shape[0]
            test_size_ratio = len(self.test_ids)/self.data.shape[0]
            if self.with_calibration:
                calib_size_ratio = len(self.calibration_ids)/self.data.shape[0]
                if self.verbose:
                    log.warning(
                        f"Train size: {train_size_ratio:.2f}, Test size: {test_size_ratio:.2f}, Calibration size: {calib_size_ratio:.2f}")
            else:
                if self.verbose:
                    log.warning(
                        f"Train size: {train_size_ratio:.2f}, Test size: {test_size_ratio:.2f}, Calibration size: 0.0")

    def __generate_exclusive_split__(self, df, test_size, seed):
        """
        Split the data into training and testing sets: makes sure that objects from train set do not appear in test set.
        """
        rnd = np.random.default_rng(seed)
        train_objects = df[["object_1", "object_2"]].drop_duplicates().sample(
            frac=1. - test_size, random_state=seed).to_numpy().flatten()

        train_mask = df["object_1"].isin(
            train_objects) | df["object_2"].isin(train_objects)
        df_train = df[train_mask]
        df_test = df[~train_mask]
        while True:
            if df_test.shape[0]/df.shape[0] > test_size:
                break
            train_objects = rnd.choice(train_objects, size=int(
                len(train_objects)*0.975), replace=False)
            train_mask = df["object_1"].isin(
                train_objects) | df["object_2"].isin(train_objects)
            df_train = df[train_mask]
            df_test = df[~train_mask]
        return df_train, df_test

    def __data_split__(self, test_size: float = 0.2, calibration_size: float = 0.2, seed: int = 42, shuffle: bool = True):
        np.random.seed(seed)
        ids = np.arange(len(self.data))
        mask = np.random.rand(len(self.data)) < 1 - test_size
        self.train_ids = ids[mask]
        self.test_ids = ids[~mask]
        if shuffle:
            np.random.seed(seed + 1)

            np.random.shuffle(self.train_ids)
            np.random.shuffle(self.test_ids)
        if self.calibration_ids:
            mask = np.random.rand(len(self.train_ids)) < 1 - calibration_size
            self.train_ids, self.calibration_ids = self.train_ids[mask], self.train_ids[~mask]
            if shuffle:
                np.random.shuffle(self.train_ids)
                np.random.shuffle(self.calibration_ids)
        else:
            self.calibration_ids = None

    def get_num_layers(self):
        return len(glob(f"outputs/activations/{self.model}/{self.datasets[0]}/{self.activation_type}/*_e.npy"))

    def get_activations(self, layer_id: int, module: str = "e"):
        """
        Get the activations for the given layer (all dataset)
        Args:
            layer_id (int): The layer id to get the activations for.
            module (str): The module to get the activations for (a - attention output, m - mlp output, e - encoder output)
        """
        assert module in [
            "a", "m", "e"], "Module must be either 'a' (self-attention), 'm' (mlp layer) or 'e' (encoder output)."
        activations = list()

        for dataset in self.datasets:
            data_dir = f"{self.activations_path}/{self.model}/{dataset}/{self.activation_type}/"
            try:
                shape = shape_as_tuple(np.load(data_dir + "shape.npy"))
                acts = np.memmap(
                    data_dir + f'layer_{layer_id}_{module}_temp.npy', shape=shape, mode="r", dtype=np.float16)
            except:
                acts = self._load_npz(
                    data_dir + f'layer_{layer_id}_{module}.npz')

            activations.append(torch.from_numpy(np.array(acts)))

        if self.activation_type == "full":
            output = stack_tensors(activations)
        else:
            raise NotImplementedError
        self._validate_activations(output)

        return output

    def _validate_activations(self, activations):
        if self.activation_type == "full":
            n = len(activations)
        else:
            n = activations.shape[0]

        n_rows = self.data.shape[0]
        assert n == n_rows, f"Number of rows in activations ({n}) does not match the number of rows in the data ({n_rows})."

    def _load_npz(self, path: str):
        return np.load(path)["arr_0"]

    def get_att_mask(self):
        masks = []
        for dataset in self.datasets:
            data_dir = f"{self.activations_path}/{self.model}/{dataset}/{self.activation_type}/mask.npy"
            masks.append(torch.from_numpy(np.load(data_dir)))
        return torch.vstack(masks)

    def get_train_att_mask(self):
        ids = self.train_ids
        return self.get_att_mask()[ids]

    def get_test_att_mask(self):
        ids = self.test_ids
        return self.get_att_mask()[ids]

    def get_cal_att_mask(self):
        ids = self.calibration_ids
        return self.get_att_mask()[ids]

    def get_dataframe(self):
        """
        Get the data (from the csv files).
        """
        return self.data

    def get_train_df(self):
        """
        Get the training data (from the csv files).
        """
        return self.data.iloc[self.train_ids.tolist()]

    def train_labeled(self, layer_id: int = -1):
        '''
        Get the train data with activation and "correct" label for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
        Returns:
            dict: A dictionary containing the embeddings and the correct labels keys:[embeddings, correct].
        '''
        correct = self.get_train_df()["correct"].to_numpy()

        if self.activation_type == "full":
            return {
                "embeddings": self.get_train_acts(layer_id=layer_id)[:, -1],
                "correct": correct,
            }
        elif self.activation_type == "last":
            return {
                "embeddings": self.get_train_acts(layer_id=layer_id),
                "correct": correct,
            }
        else:
            raise NotImplementedError

    def train_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        ''' 
        Get the training bags for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            drop_zeros (bool): Whether to drop padded rows from the bags.
        Returns:
            dict: A dictionary containing the bags, the correct labels and the last embedding keys:[embeddings, correct, last_embedding].
        '''
        assert self.activation_type == "full", "Bags can only be generated for full activations."
        correct = self.get_train_df()["correct"].to_numpy()
        acts = self.get_train_acts(layer_id=layer_id)
        try:
            mask = self.get_train_att_mask()
        except:
            mask = None
        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts
        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }

    def test_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        ''' 
        Get the test bags for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            drop_zeros (bool): Whether to drop padded rows from the bags.
        Returns:
            dict: A dictionary containing the bags, the correct labels and the last embedding keys:[embeddings, correct, last_embedding].
        '''
        assert self.activation_type == "full", "Bags can only be generated for full activations."
        correct = self.get_test_df()["correct"].to_numpy()
        acts = self.get_test_acts(layer_id=layer_id)
        try:
            mask = self.get_test_att_mask()
        except:
            mask = None
        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts
        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }

    def _drop_zeros(self, acts, mask: None):
        for bag in acts:
            if bag.shape[0] == 0:
                print("Bag is empty BEFORE THE DROP ZEROS")
        if mask is not None:
            bags = [drop_zero_rows(self._drop_zeros_eisum(i, m))
                    for i, m in zip(acts, mask)]
        elif mask is None:
            log.warning("No mask found. Not dropping zeros without mask.")
        for bag in bags:
            if bag.shape[0] == 0:
                print("Bag is empty AFTER THE DROP ZEROS")
        return bags

    def _drop_zeros_eisum(self, act, mask):
        if act.shape[0] == mask.shape[0]:
            return torch.einsum('lh, l -> lh', act, mask)
        else:
            shape = mask.shape[0]
            mask[-5:] = 1
            output = torch.einsum('lh, l -> lh', act[-shape:], mask)
            return output

    def cal_bags(self, layer_id: int = -1, drop_zeros: bool = True):
        ''' 
        Get the calibration bags for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            drop_zeros (bool): Whether to drop padded rows from the bags.
        Returns:
            dict: A dictionary containing the bags, the correct labels and the last embedding keys:[embeddings, correct, last_embedding].
        '''
        assert self.activation_type == "full", "Bags can only be generated for full activations."
        correct = self.get_cal_df()["correct"].to_numpy()
        acts = self.get_cal_acts(layer_id=layer_id)
        try:
            mask = self.get_cal_att_mask()
        except:
            mask = None
        if drop_zeros:
            bags = self._drop_zeros(acts, mask)
        else:
            bags = acts
        return {
            "embeddings": bags,
            "correct": correct,
            "last_embedding": acts[:, -1],
        }

    def test_labeled(self, layer_id: int = -1):
        '''
        Get the test data with activation and "correct" label for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
        Returns:
            dict: A dictionary containing the embeddings and the correct labels keys:[embeddings, correct].
        '''
        correct = self.get_test_df()["correct"].to_numpy()

        if self.activation_type == "full":
            return {
                "embeddings": self.get_test_acts(layer_id=layer_id)[:, -1],
                "correct": correct,
            }
        elif self.activation_type == "last":
            return {
                "embeddings": self.get_test_acts(layer_id=layer_id),
                "correct": correct,
            }
        else:
            raise NotImplementedError

    def cal_labeled(self, layer_id: int = -1):
        '''
        Get the calibration data with activation and "correct" label for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
        Returns:
            dict: A dictionary containing the embeddings and the correct labels keys:[embeddings, correct].
        '''
        correct = self.get_cal_df()["correct"].to_numpy()

        if self.activation_type == "full":
            return {
                "embeddings": self.get_cal_acts(layer_id=layer_id)[:, -1],
                "correct": correct,
            }
        elif self.activation_type == "last":
            return {
                "embeddings": self.get_cal_acts(layer_id=layer_id),
                "correct": correct,
            }
        else:
            raise NotImplementedError

    def get_train_scores(self, layer_id: int = -1):
        """
        Get the training scores.
        """
        if layer_id == -1:
            return self.data.iloc[self.train_ids]["scores"].to_numpy()
        else:
            return self.data.iloc[self.train_ids][f"scores_{layer_id}"].to_numpy()

    def get_test_scores(self, layer_id: int = -1):
        """
        Get the testing scores.
        """
        if layer_id == -1:
            return self.data.iloc[self.test_ids]["scores"].to_numpy()
        else:
            return self.data.iloc[self.test_ids][f"scores_{layer_id}"].to_numpy()

    def get_cal_scores(self, layer_id: int = -1):
        """
        Get the calibration scores.
        """
        if layer_id == -1:
            return self.data.iloc[self.calibration_ids]["scores"].to_numpy()
        else:
            return self.data.iloc[self.calibration_ids][f"scores_{layer_id}"].to_numpy()

    def get_test_df(self):
        """
        Get the testing data (from the csv files).
        """
        return self.data.iloc[self.test_ids.tolist()]

    def get_cal_df(self):
        """
        Get the calibration data (from the csv files).
        """
        return self.data.iloc[self.calibration_ids.tolist()]

    def get_train_acts(self, layer_id: int, module: str = "e"):
        """
        Get the training activations for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            module (str): The module to get the activations for (a - attention output, m - mlp output, e - encoder output)
        """
        return self.get_activations(layer_id, module)[self.train_ids]

    def get_test_acts(self, layer_id: int, module: str = "e"):
        """
        Get the testing activations for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            module (str): The module to get the activations for (a - attention output, m - mlp output, e - encoder output)
        """
        return self.get_activations(layer_id, module)[self.test_ids]

    def get_cal_acts(self, layer_id: int, module: str = "e"):
        """
        Get the calibration activations for the given layer.
        Args:
            layer_id (int): The layer id to get the activations for.
            module (str): The module to get the activations for (a - attention output, m - mlp output, e - encoder output)
        """
        return self.get_activations(layer_id, module)[self.calibration_ids]

    def column_list(self):
        """
        Get the list of columns in the datasets.
        """
        columns = set()
        for dataset in self.datasets:
            lf = pl.scan_csv(
                f"{self.dataset_path}{dataset}.csv")
            try:
                columns.update(lf.collect_schema().names())
            except:
                columns.update(lf.columns)
        return list(columns)


def drop_zero_rows(X):
    """
    Drop rows with all zeros in a matrix.
    """
    return X[(X != 0).any(axis=1)]


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
