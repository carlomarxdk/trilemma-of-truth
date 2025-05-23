import logging
from typing import List, Dict, Callable
from abc import ABC, abstractmethod
log = logging.getLogger(__name__)

LEGAL_TASKS = ["binary", "multichoice"]
LEGAL_TYPES = ["default", "masked", "instruct"]


class PromptTemplate(ABC):

    def __init__(self, task: int = 0, prompt_type: str = "default", enumeration: list = None,
                 system_role: str = "system", user_role: str = "user", assist_role: str = "assistant"):
        assert prompt_type in LEGAL_TYPES, \
            f"Prompt type must be one of {LEGAL_TYPES}"
        self.task = task
        self.prompt_type = prompt_type
        self.enumeration = self.format_enumeration_list(enumeration)
        # Used only for the instruct template
        self.system_role = system_role
        self.user_role = user_role
        self.assist_role = assist_role

    def format_enumeration_list(self, enumeration: List) -> List[str]:
        return [str(e) for e in enumeration]

    def __call__(self, statement: str):
        return self.get_prompt(statement)

    @property
    @abstractmethod
    def enum_unit(self) -> str:
        raise NotImplementedError()

    @property
    def templates(self) -> Dict[str, Callable[[str], str | list[dict]]]:
        return {
            "default":
            lambda x: f"Question: Is the following statement correct? {x}\n\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n\n"
                f"Please respond with the corresponding {self.enum_unit}. "
                f"The final answer is {self.enum_unit} ",
            "masked":
            lambda x: f"Question: Is the following statement correct? {x}\n\n"
                "Select one of the following options:\n"
                f"{self.options_as_str()}\n\n"
                f"Please respond with the corresponding {self.enum_unit}. "
                f"The final answer is {self.enum_unit} [MASK].",
            "instruct":
                lambda x: self.instruct_template(x),
        }

    def return_intro(self):
        baseline = 'Three plus three equals six.'
        if self.prompt_type == 'default':
            prompt = self.get_prompt(baseline)
            return prompt.split(baseline)[0]
        elif self.prompt_type == 'instruct':
            prompt = self.get_prompt(baseline)
            prompt_baseline = []
            for i in range(len(prompt)):
                prompt_baseline.append(prompt[i])
                if baseline in prompt[i]['content']:
                    prompt_baseline[i]['content'] = prompt[i]['content'].split(baseline)[
                        0]
                    break
            return prompt_baseline

    def options_as_dict(self) -> Dict[str, str]:
        return {str(e): a for e, a in zip(self.enumeration, self.answers)}

    def options_as_str(self) -> str:
        return "\n".join([f"{e}. {a}" for e, a in self.options_as_dict().items()])

    @property
    @abstractmethod
    def question_type(self) -> str:
        raise NotImplementedError()

    def get_prompt(self, statement: str) -> str | list[dict]:
        assert isinstance(statement, str), "Statement must be a string."
        assert len(
            statement) > 2, "Statement is too short. Please provide a longer statement."
        return self.templates[self.prompt_type](statement)

    def instruct_template(self, statement: str):

        prompt = f"Question: Is the following statement correct? {statement}\n\n Select one of the following options:\n{self.options_as_str()}\n"
        if self.system_role == self.user_role:
            return [
                {'role': f'{self.user_role}',
                 "content": f"You are an expert in fact-checking. Your task is to assist the user by answering questions based on your comprehensive knowledge. Please respond with the corresponding {self.enum_unit}.\n\n{prompt}"},
                {'role': f'{self.assist_role}',
                 'content': f"The final answer is {self.enum_unit} "}
            ]
        else:
            return [
                {'role': f'{self.system_role}',
                 "content": "You are an expert in fact-checking. Your task is to assist the user by answering questions based "
                 f"on your comprehensive knowledge. Please respond with the corresponding {self.enum_unit}."},
                {'role': f'{self.user_role}',
                 'content': f'{prompt}'},
                {'role': f'{self.assist_role}',
                 'content': f"The final answer is {self.enum_unit} "}
            ]

    @property
    @abstractmethod
    def answers(self) -> List[str]:
        raise NotImplementedError()

    @property
    def num_answers(self) -> int:
        return len(self.answers)

    @property
    def task(self) -> int:
        return self._task

    @task.setter
    def task(self, task: int) -> None:
        assert task in [
            0, 1], "Task must be 0 for False -> Truth or 1 for Null->Know"
        self._task = task

    @property
    def enumeration(self) -> List[str]:
        return self._enumeration

    @enumeration.setter
    def enumeration(self, enumeration: List[str]) -> None:
        assert len(enumeration) == len(
            self.answers), "The number of options must match the number of answers."
        self._enumeration = enumeration


class BinaryPrompt(PromptTemplate):
    def __init__(self, task: int = 0, prompt_type: str = "default", enumeration: list = ["1", "2"],
                 system_role: str = "system", user_role: str = "user", assist_role: str = "assistant"):
        super().__init__(task=task, prompt_type=prompt_type, enumeration=enumeration, system_role=system_role,
                         user_role=user_role, assist_role=assist_role)

    @property
    def question_type(self):
        return "binary"

    @property
    def enum_unit(self):
        return "number"

    @property
    def answers(self):
        return ["Yes", "No"]


class MultichoicePrompt(PromptTemplate):
    def __init__(self, task: int = 0, prompt_type: str = "default", enumeration: list = ["1", "2", "3", "4", "5", "6"],
                 system_role: str = "system", user_role: str = "user", assist_role: str = "assistant"):
        super().__init__(task=task, prompt_type=prompt_type, enumeration=enumeration, system_role=system_role,
                         user_role=user_role, assist_role=assist_role)

    @property
    def question_type(self):
        return "multichoice"

    @property
    def enum_unit(self):
        return "number"

    @property
    def answers(self):
        return ["The statement is correct",
                "The statement is incorrect",
                "I do not have sufficient knowledge about the statement",
                "The statement is too ambiguous to provide a reliable answer",
                "All of the above options are correct",
                "None of the above options are applicable"]


class MultichoicePromptTF(MultichoicePrompt):
    @property
    def answers(self):
        return ["The statement is true",
                "The statement is false",
                "I do not have sufficient knowledge about the statement",
                "The statement is too ambiguous to provide a reliable answer",
                "All of the above options are correct",
                "None of the above options are applicable"]


class MultichoicePromptABC(MultichoicePrompt):
    def __init__(self, task: int = 0, prompt_type: str = "default", enumeration: list = ["A", "B", "C", "D", "E", "F"],
                 system_role: str = "system", user_role: str = "user", assist_role: str = "assistant"):
        super().__init__(task=task, prompt_type=prompt_type, enumeration=enumeration, system_role=system_role,
                         user_role=user_role, assist_role=assist_role)

    @property
    def enum_unit(self):
        return "the uppercase letter"
