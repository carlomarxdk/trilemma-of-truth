from abc import ABC, abstractmethod
from typing import Dict, List
import random
import numpy as np
import polars as pl


class DatasetGenerator(ABC):
    def __init__(self, source: Dict[str, List[str]],
                 random_seed: int = 42,
                 is_fake: bool = False, category=None):
        self.source = source
        self.random_seed = random_seed
        self.keys = list(set(source.keys()))
        self.values = self._get_values()

        self.data = None
        self.is_fake = is_fake
        assert category is not None
        self.category = category

    @abstractmethod
    def apply_template(self, negated=False, **kwargs):
        NotImplemented

    def _get_values(self):
        values = set()
        for v in self.source.values():
            values.update(v)
        return values

    def lookup_incorrect(self, key) -> str:
        '''
        Return False condition for a given drug
        '''
        correct = self.source[key]
        choice = random.choice(list(set(self.values) - set(correct)))
        if choice.lower() in [c.lower() for c in correct]:
            return self.lookup_incorrect(key)
        # elif abbreviate(choice) in [abbreviate(c) for c in correct] or abbreviate(choice) in [c for c in correct]:
        #     return self.lookup_incorrect(key)
        elif any(word in c.lower().split() for word in choice.lower().split() for c in correct):
            return self.lookup_incorrect(key)
        return choice

    def generate_sample(self, key, value, negated: bool):
        correct_values = self.source[key]
        correct = any([value.lower() in v.lower() for v in correct_values])
        if negated:
            correct = not correct
        if not self.is_fake:
            return {'statement': self.apply_template(key, value, negated),
                    'object_1': key,
                    'object_2': value,
                    'correct_object_2':  correct_values,
                    'correct': correct,
                    'negation': negated,
                    'real_object': True,
                    # 'fake_object': False,
                    # 'fictional_object': False,
                    'category': self.category
                    }
        else:
            return {'statement': self.apply_template(key, value, negated),
                    'object_1': key,
                    'object_2': value,
                    # correct_values do notmean anything in this case
                    'correct_object_2':  correct_values,
                    'correct': False,
                    'negation': negated,
                    'real_object': False,
                    # 'fake_object': True,
                    # 'fictional_object': False,
                    'category': self.category
                    }

    def generate_full_dataset(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        output = []
        for key in self.keys:
            for value in self.source[key]:
                output.append(self.generate_sample(key, value, negated=True))
                output.append(self.generate_sample(key, value, negated=False))
                incorrect = self.lookup_incorrect(key)
                output.append(self.generate_sample(
                    key, incorrect, negated=True))
                incorrect = self.lookup_incorrect(key)
                output.append(self.generate_sample(
                    key, incorrect, negated=False))
        self.data = pl.from_records(output)
        return self.data

    def generate_subsample(self, n: int = 5000, seed: int = 42):
        return self.data.sample(n, seed=seed, shuffle=True)



class CityCountry(DatasetGenerator):
    '''
    Class to handle the dataset from DrugBank
    '''    
    def apply_template(self, city: str, country: str, negated: bool=False):
        if 'city' in city.lower():
            if negated:
                return f"{city} is not located in {country}."
            else:
                return f"{city} is located in {country}."
        else:
            if negated:
                return f"The city of {city} is not located in {country}."
            else:
                return f"The city of {city} is located in {country}."
    
        
    def add_population(self, population_dict: dict):
        self.data = self.data.with_columns(
            pl.col("object_1").map_elements(lambda x: population_dict[x], return_dtype=pl.Int32).alias("max_population")
        )
        return self.data

        
    def generate_subsample(self, n: int, seed: int, objects: list = None):
        np.random.seed(seed)
        if objects is not None:
            data = self.data.filter(pl.col("object_1").is_in(objects))
        else:
            data = self.data
        if data.height > n:
            print(f'Downsample from {data.height} to {n}')
            data = data.sample(n, seed=seed, shuffle=True)
        else:
            print(f'Size of the dataset is {data.height}')
        return data



##### Drug-Disease Dataset #####
def abbreviate(entity):
    return ''.join([i[0] for i in entity.split()]).upper()

class DrugDisease(DatasetGenerator):
    '''
    Class to handle the dataset from DrugBank
    '''
    def apply_template(self, drug: str, condition: str, negated: bool=False):
        if negated:
            return f"{drug} is not indicated for the treatment of {condition}."
        else:
            return f"{drug} is indicated for the treatment of {condition}."
        
        

    def lookup_incorrect(self, key) -> str:
        '''
        Return False condition for a given drug
        '''
        correct = self.source[key]
        choice = random.choice(list(set(self.values) - set(correct)))
        if choice.lower() in [c.lower() for c in correct]:
            return self.lookup_incorrect(key)
        elif abbreviate(choice) in [abbreviate(c) for c in correct] or abbreviate(choice) in [c for c in correct]:
            return self.lookup_incorrect(key)
        elif any(word in c.lower().split() for word in choice.lower().split() for c in correct):
            return self.lookup_incorrect(key)
        return choice