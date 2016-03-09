# -*- coding: utf-8 -*-

from weight import model
import pandas as pd


class TestModel:

    # Pytest Fixtures

    def setup(self):
        print("setup             class:TestStuff")

    def teardown(self):
        print("teardown          class:TestStuff")

    def setup_class(cls):
        print("setup_class       class:%s" % cls.__name__)

    def teardown_class(cls):
        print("teardown_class    class:%s" % cls.__name__)

    def setup_method(self, method):
        print("setup_method      method:%s" % method.__name__)

    def teardown_method(self, method):
        print("teardown_method   method:%s" % method.__name__)

    # Test Cases

    def test_construct(self):
        m = model.Model()
        assert(m is not None)

    def test_select(self):
        m = model.Model()

        enddate = m.enddate()
        prev_week = enddate - pd.DateOffset(weeks=1)

        df = m.select(prev_week)
        assert(df is not None)


#
# References:
#   http://pythontesting.net/framework/pytest/pytest-introduction/

#
# Test Discovery
#   Name your test modules/files starting with ‘test_’.
#   Name your test functions starting with ‘test_’.
#   Name your test classes starting with ‘Test’.
#   Name your test methods starting with ‘test_’.
#   Make sure all packages with test code have an ‘__init.py__’ file.

#
# Running Pytest
#   py.test
#   py.test -v          [verbose]
#   py.test -s          [Turn off output capture]

#
# Pytest Fixtures:
#   setup_module/teardown_module
#       Called at the beginning and end of a module of test code
#   setup_function/teardown_function
#       Called before and after a test function call
#   setup_class/teardown_class
#       Called at the beginning and end of a class of test methods
#   setup/teardown
#       Alternate style of the class level fixtures
#   setup_method/teardown_method
#       Called before and after a test method call
