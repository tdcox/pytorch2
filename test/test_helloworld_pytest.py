import helloworld


# Using py.test framework

def test_hello():
    assert helloworld.hello() == 'Hello, World!'
