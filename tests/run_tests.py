#!/usr/bin/env python
"""
Test runner script to execute all tests for the NBA player minutes predictor.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == '__main__':
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern='test_*.py')
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful()) 