from datetime import datetime
from typing import Dict, List

from data_reader import DataPoint


def boolean_string(s: str) -> bool:
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


def get_current_time() -> str:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time


def compute_dict_average(dict: Dict) -> float:
    # empty dictionary, return 0
    if not dict:
        return 0

    total = 0
    N = 0
    for key, value in dict.items():
        total += value
        N += 1
    return total / N


def check_test_alignment(
    test_inputs: Dict[str, str], test_info: Dict[str, List[DataPoint]]
) -> None:
    for warning in test_inputs:
        inputs = test_inputs[warning]
        infos = test_info[warning]
        for i, code in enumerate(inputs):
            assert code == infos[i].GetT5Representation(True)[0], "something wrong! stop it!"
