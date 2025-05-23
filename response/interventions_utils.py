import re
import numpy as np
import torch
import torch
import torch.nn as nn
import numpy as np


def torch_list_to_numpy(x, eps=1e-6):
    ''' 
    Transform a list of torch tensor items into a numpy arra
    Args;
        x: List of N tensors of various lengths
    Returns:
        numpy.array of size N

    '''
    _x = []
    for i in x:
        _x.append(i.sum() + eps)
    return np.array([p.cpu() for p in _x])


def to_log_proba(proba_list):
    '''
    Transform an array of probabilities into a marginal log-proba
    Args:
        proba_list: numpy.array of size N with probabilties
    Returns:
        marginal log-proba
    '''
    return np.sum(np.log(torch_list_to_numpy(proba_list)))


class InterventionDataProcessor:
    '''
    Class that handles data for interventions
    '''

    def __init__(self, datahandler, tokenizer, datapack_name):
        '''Initialize the class
        Args:
            datahandler: DataHandler object
            tokenizer: Tokenizer object
            datapack_name: str, name of the datapack
        '''
        self.dh = datahandler
        self.datapack = datapack_name
        self.tokenizer = tokenizer

    def template(self, object_1, object_2, negation, category=None):
        ''' 
        Apply template 
        Args:
            object_1: str, first object
            object_2: str, second object
            negation: int, 0 or 1
        Returns:
            str, formatted statement
        '''
        article = "is" if negation == 0 else "is not"
        if self.datapack in ['cities', 'cities_loc']:
            if 'city' in object_1.lower():
                return f'{object_1} is located in'
            return f'The city of {object_1} {article} located in'
        elif self.datapack in ['drugs', 'med_indications']:
            if any(word in object_1.lower() for word in ["control", "preparation", "contraception", "prevention", "weight loss"]):
                return f'{object_1.capitalize()} {article} indicated for the treatment of'
            return f'{object_1.capitalize()} {article} indicated for the treatment of'
        elif self.datapack == 'symptoms':
            return f'{object_1.capitalize()} {article} linked to'
        elif self.datapack in ['definitions', 'defs']:
            if category == 'instances':
                return f'{object_1} {article} a'
            elif category == 'synonyms':
                return f'{object_1} {article} a synonym of a'
            elif category == 'types':
                return f'{object_1} {article} a type of a'
            else:
                return f'{object_1} {article} a'
        else:
            raise ValueError("Invalid data pack")

    def return_processed_test_df(self):
        test_data = self.dh.get_test_df(
        )[['object_1', 'object_2', 'correct_object_2', 'real_object', 'correct', 'negation', 'category']]

        test_data['answer'] = test_data['object_2']
        test_data['statement'] = test_data.apply(
            lambda row: self.template(row['object_1'], row['object_2'], row['negation'], row['category']), axis=1)

        return test_data

    def get_answer_ids(self, answer):
        return self.tokenizer(
            answer,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids[0]

    def get_answer_seq_ids(self, statement, answer):
        """ 
        For long answers
        """
        answers = [' ' + a.rstrip() for a in answer.split(' ')]
        answers_ids = []
        current = statement
        if current[-1] == ' ':
            current = current.rstrip()
        statements = [current]
        init_statement_ids = self._statement_to_ids(current)

        for a in answers:
            current += a
            answers_ids.append(self._answer_to_ids(a))
            statements.append(current)
        return statements, answers, answers_ids, init_statement_ids

    def _statement_to_ids(self, statement):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(statement))

    def _answer_to_ids(self, answer):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))


class InstructInterventionDataProcessor(InterventionDataProcessor):
    def __init__(self, datahandler, tokenizer, datapack_name, user_role, system_role, assist_role):
        super().__init__(datahandler, tokenizer, datapack_name)
        # Used only for the instruct template
        self.system_role = system_role
        self.user_role = user_role
        self.assist_role = assist_role

    def _instruct_template(self, statement: str):
        if self.system_role == self.user_role:
            return [
                {'role': f'{self.user_role}',
                 "content": f"You are an expert in fact-checking. Complete this statement: {statement}"},
            ]
        else:
            return [
                {'role': f'{self.system_role}',
                 "content": "You are an expert in fact-checking. Complete the statement provided by the user."},
                {'role': f'{self.user_role}',
                 'content': f'{statement}'},
            ]

    # def _template(self, object_1, negation, category=None):
    #     '''
    #     Apply template
    #     Args:
    #         object_1: str, first object
    #         object_2: str, second object
    #         negation: int, 0 or 1
    #     Returns:
    #         str, formatted statement
    #     '''
    #     article = "is" if negation == 0 else "is not"
    #     if self.datapack in ['cities', 'cities_loc']:
    #         if 'city' in object_1.lower():
    #             return f'{object_1} is located in'
    #         return f'The city of {object_1} {article} located in'
    #     elif self.datapack in ['drugs', 'med_indications']:
    #         if any(word in object_1.lower() for word in ["control", "preparation", "contraception", "prevention", "weight loss"]):
    #             return f'{object_1.capitalize()} {article} used for'
    #         return f'{object_1.capitalize()} {article} is indicated for the treatment of'
    #     elif self.datapack == 'symptoms':
    #         return f'{object_1.capitalize()} {article} linked to'
    #     elif self.datapack in ['definitions', 'defs']:
    #         if category == 'instances':
    #             return f'{object_1} {article} a'
    #         elif category == 'synonyms':
    #             return f'{object_1} {article} a synonym of a'
    #         elif category == 'types':
    #             return f'{object_1} {article} a type of a'
    #         else:
    #             return f'{object_1} {article}'
    #     else:
    #         raise ValueError("Invalid data pack")

    def _template(self, object_1, object_2, negation, category=None):
        statement = self.template(object_1, object_2, negation, category)
        return self._instruct_template(statement)

    def return_processed_test_df(self):
        test_data = self.dh.get_test_df(
        )[['object_1', 'object_2', 'correct_object_2', 'real_object', 'correct', 'negation', 'category']]
        test_data['answer'] = test_data['object_2']
        test_data['statement'] = test_data.apply(
            lambda row: self._template(row['object_1'], row['object_2'], row['negation'], row['category']), axis=1)
        return test_data

    def get_answer_ids(self, answer):
        return self.tokenizer(
            answer,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids[0]

    def get_answer_seq_ids(self, statement, answer):
        """ 
        For long answers
        """
        answers = [' ' + a.rstrip() for a in answer.split(' ')]

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
        return self.tokenizer.apply_chat_template(statement, add_generation_prompt=True, tokenize=False)

    def _statement_to_ids(self, statement):
        return self.tokenizer.apply_chat_template(statement, add_generation_prompt=False, tokenize=True)


class TorchStandardScaler(nn.Module):
    def __init__(self, mean, scale):
        super(TorchStandardScaler, self).__init__()
        # Store mean and scale as PyTorch tensors for efficient computation
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)

    @classmethod
    def from_sklearn(cls, scaler):
        """
        Create a TorchStandardScaler from a scikit-learn StandardScaler.
        """
        return cls(scaler.mean_, scaler.scale_)

    def transform(self, X):
        """
        Apply standard scaling to the input tensor.
        """
        return (X - self.mean) / self.scale

    def inverse_transform(self, X_scaled):
        """
        Reverse the standard scaling transformation.
        """
        return X_scaled * self.scale + self.mean


class Attribution:
    def reduce(self, x):
        if self.reduction == "sum":
            return x.sum()
        elif self.reduction == "mean":
            return x.mean()
        elif self.reduction == "max":
            return x.max()
        else:
            raise ValueError("Invalid reduction")

    def score_single(self, x):
        raise NotImplementedError

    def __call__(self, x):
        if x.ndim == 2:
            return self.score_single(x)
        elif x.ndim == 3:
            return self.score_single(x[0])


class TCAV(Attribution):
    def __init__(self, concept_direction, reduction="sum", seed: int = 2024, n_random_directions: int = 1000):
        """
        Initialize TCAV attribution method
        Args:
            concept_direction: Concept direction
            reduction: Reduction method
            seed: Random seed
            n_random_directions: Number of random directions to sample
        """
        self.dir = concept_direction
        self.reduction = reduction
        self.seed = seed
        self.n_random = n_random_directions
        self.rdirs = self.random_direction(
            concept_direction, n_random_directions)

    def score_single(self, x):
        assert x.ndim == 2
        norm = x.norm(dim=1) ** -1
        x = torch.einsum("lh, l -> lh", x, norm)
        sr = torch.einsum("lh, hr -> rl", x, self.rdirs)
        return {"default": self.reduce(torch.einsum("lh, h -> l", x, self.dir)),
                "random": torch.stack([self.reduce(i) for i in sr])}

    def random_direction(self, direction, n_random):
        torch.manual_seed(self.seed)
        output = torch.zeros((direction.shape[0], n_random))
        for i in range(n_random):
            output[:, i] = direction.clone(
            )[torch.randperm(direction.shape[0])]

        return output.to(direction.device)


def translate_concept(X, direction, target_coord: float, absolute=False):
    """
    Translate the 'X' embedding in the specified direction.

    If absolute is False (default), the embedding is moved so that its
    projection along 'direction' becomes 'target_coord'.
    If absolute is True, the embedding is moved by 'target_coord' units
    along 'direction'.

    Args:
    - X: The embedding to translate (tensor of shape [B, S, H])
    - direction: The direction to translate the embedding (tensor of shape [H])
    - target_coord: 
         - If absolute is False: the new coordinate to set along the direction.
         - If absolute is True: the number of units to move along the direction.
    - absolute: If True, move by `target_coord` units, else move so that the
                new coordinate is `target_coord`.

    Returns:
    - The translated embedding (tensor with same shape as X)
    """
    # Normalize the translation direction
    unit_dir = direction / torch.norm(direction)

    # Compute the current projection of X along the direction.
    # This yields a tensor of shape [B, S]
    curr_coord = torch.einsum("bsh, h -> bs", X, unit_dir)

    # Compute the translation delta based on the mode
    if absolute:
        # Move by target_coord units along the direction
        delta = torch.full_like(curr_coord, fill_value=target_coord)
    else:
        # Move so that the new coordinate becomes target_coord
        delta = torch.full_like(
            curr_coord, fill_value=target_coord) - curr_coord

    # Expand delta into the embedding space along the feature dimension
    translation = torch.einsum("h, bs -> bsh", unit_dir, delta)

    # Translate the original embedding
    X_translated = X + translation
    return X_translated


def amplify_concept(X, direction, scaler: float = None):
    if torch.norm(direction) != 1:
        direction = direction / torch.norm(direction)
    curr_coord = torch.einsum("bsh, h -> bs", X, direction)
    step = torch.sign(curr_coord) * scaler
    proj = torch.einsum("h, bs -> bsh", direction, step)
    Xs = X + proj
    return Xs


def polarize_concept(X, direction, scaler: float = None, positive=True):
    if torch.norm(direction) != 1:
        direction = direction / torch.norm(direction)
    curr_coord = torch.einsum("bsh, h -> bs", X, direction)
    if positive:
        step = (torch.sign(curr_coord) * scaler)
        step[step < 0] = 0
    else:
        step = (torch.sign(curr_coord) * scaler)
        step[step > 0] = 0
    proj = torch.einsum("h, bs -> bsh", direction, step)
    Xs = X + proj
    return Xs


def zero_out_concept(X, direction, scaler: float = None):
    non_zero = (direction == 0).float()
    Xs = torch.einsum("bsh, h -> bsh", X, non_zero)
    return Xs


def indirect_effect(p, p_new, targets):
    res = {}
    for target in targets:
        diff = np.array(p[:, target] - p_new[:, target])
        r = (np.mean(diff))/(diff.std(ddof=1)/np.sqrt(len(diff)))
        res[target] = r
    return res


def NIE_ft(p, p_new, labels):
    pd_minus = ((p[:, 0] - p[:, 1])[labels == 0]).mean()
    pd_plus = ((p[:, 0] - p[:, 1])[labels == 1]).mean()
    pd_minus_star = ((p_new[:, 0] - p_new[:, 1])[labels == 0]).mean()
    return (pd_minus_star - pd_minus)/(pd_plus - pd_minus)


def NIE_tf(p, p_new, labels):
    pd_minus = ((p[:, 0] - p[:, 1])[labels == 0]).mean()
    pd_plus = ((p[:, 0] - p[:, 1])[labels == 1]).mean()
    pd_plus_star = ((p_new[:, 0] - p_new[:, 1])[labels == 1]).mean()
    return (pd_plus_star - pd_plus)/(pd_minus - pd_plus)


def decoherence(p, p_new, valid_classes):
    p_valid = p[:, valid_classes].sum(1)
    p_new_valid = p_new[:, valid_classes].sum(1)
    p_invalid = 1 - p_valid
    p_new_invalid = 1 - p_new_valid
    return (p_new_invalid/p_invalid)


def localized_effect_ratio(p, p_new, invalid_classes):
    p_invalid = np.abs(p[:, invalid_classes] - p_new[:, invalid_classes]).sum()
    p_all = np.abs(p - p_new).sum()
    return p_invalid/p_all


def extract_coef_numbers(file_paths):
    """
    Extract numbers that appear after 'coef_' in a list of file paths.

    Args:
        file_paths (list): List of file paths as strings.

    Returns:
        list: List of integers representing the numbers after 'coef_'.
    """
    coef_numbers = []
    for path in file_paths:
        match = re.search(r'coef_(\d+)', path)
        if match:
            coef_numbers.append(int(match.group(1)))
    return coef_numbers


def extract_result_numbers(file_paths):
    """
    Extract numbers that appear after 'coef_' in a list of file paths.

    Args:
        file_paths (list): List of file paths as strings.

    Returns:
        list: List of integers representing the numbers after 'coef_'.
    """
    coef_numbers = []
    for path in file_paths:
        match = re.search(r'results_(\d+)', path)
        if match:
            coef_numbers.append(int(match.group(1)))
    return coef_numbers
