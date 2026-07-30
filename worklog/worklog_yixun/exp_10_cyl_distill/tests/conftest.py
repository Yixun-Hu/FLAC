import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(D, f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
