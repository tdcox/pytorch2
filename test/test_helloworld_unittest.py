import unittest

import helloworld


# Using Unittest framework

class HelloWorldTest(unittest.TestCase):
    def test_hello(self):
        self.assertEqual(helloworld.hello(), 'Hello, World!')


if __name__ == '__main__':
    unittest.main()
