from autocurricula.games.math import d_format

import pytest


@pytest.fixture
def problems():
    return ["What is 1+1?", "What is 2+2?"]


@pytest.fixture
def solutions():
    return ["1+1=2", "2+2=4"]


def test_d_format(problems, solutions):
    prompts = d_format(problems, solutions)
    assert [isinstance(prompt, str) for prompt in prompts]
