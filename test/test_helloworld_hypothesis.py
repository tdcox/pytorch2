import helloworld
from hypothesis import given
from hypothesis.strategies import text


@given(text())
def test_helloname(s):
    assert helloworld.helloname(s) == 'Hello, ' + s + '!'
