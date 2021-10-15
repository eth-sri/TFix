from datetime import datetime
from typing import Dict


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
