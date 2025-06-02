
import numpy as np
import logging
from copy import deepcopy
log = logging.getLogger('task')


LEGAL_TASKS = [-1,0,1,2,3]
class Task:
    def __init__(self, task: int = 0):
        ''' 
        Class to handle the task and assign correct labels to the data'''
        assert task in LEGAL_TASKS, f"Task must be in {LEGAL_TASKS}"
        
        if task == 0:
            log.warning("The TASK is set to: TRUE-vs-ALL")
        elif task == 1:
            log.warning("The TASK is set to: FALSE-vs-ALL")
        elif task == 2:
            log.warning("The TASK is set to: UNVERIFIABLE-vs-ALL")
        elif task == 3:
            log.warning("The TASK is set to: TRUE-vs-FALSE")
        elif task == -1:
            log.warning("The TASK is set to: MULTICLASS")
        self.task = task

    def _return_labels(self, correct, real):
        '''
        Function to return the correct labels for the task
        Args:
            correct: The correct statement (np.array)
            real: The real statement (np.array)
        Returns:
            A dictionary with the correct labels and the mask
        '''
        assert self.is_binary(correct), "The correct labels must be binary"
        assert self.is_binary(real), "The real labels must be binary"
        assert real.shape == correct.shape, "The shapes must be the same"

        correct = np.copy(correct)
        real = np.copy(real)

        if self.task == 0:
            new_targets = np.zeros_like(correct)
            new_targets[(correct == 1) & (real == 1)] = 1
            assert self.is_binary(
                new_targets), "The new targets must be binary"
            return {'targets': correct, 'mask': np.ones_like(real)}
        elif self.task == 1:
            new_targets = np.zeros_like(correct)
            new_targets[(correct == 0) & (real == 1)] = 1
            assert self.is_binary(
                new_targets), "The new targets must be binary"
            return {'targets': new_targets, 'mask': np.ones_like(real)}
        elif self.task == 2:
            new_targets = np.zeros_like(real)
            new_targets[real == 0] = 1
            assert self.is_binary(
                new_targets), "The new targets must be binary"
            return {'targets': new_targets, 'mask': np.ones_like(real)}
        elif self.task == 3:
            new_targets = np.zeros_like(correct)
            new_targets[(correct == 1) & (real == 1)] = 1
            mask = np.zeros(real)
            mask[real == 1] = 1 # only keep the real samples (not the synthetic ones)
            assert self.is_binary(
                new_targets), "The new targets must be binary"
            return {'targets': new_targets, 'mask': mask}
        elif self.task == -1:
            new_targets = np.zeros_like(correct)
            new_targets[(correct == 1) & (real == 1)] = 1
            new_targets[(real == 0)] = 2
            return {'targets': new_targets, 'mask': np.ones_like(real)}

    def return_labels(self, correct, real):
        x = self._return_labels(correct, real)
        return {'targets': deepcopy(x["targets"]),
                'mask': deepcopy(x["mask"].astype(bool))}

    def is_binary(self, x):
        return set(x).issubset({0, 1})
