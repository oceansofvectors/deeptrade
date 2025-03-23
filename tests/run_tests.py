#!/usr/bin/env python
import unittest
import sys
import os

# Add parent directory to path so we can import modules from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from test_risk_manager import TestRiskManager
from test_trade_with_risk_management import TestTradeWithRiskManagement
from test_money_integration import TestMoneyIntegration

if __name__ == "__main__":
    # Create a test loader
    loader = unittest.TestLoader()
    
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTests(loader.loadTestsFromTestCase(TestRiskManager))
    test_suite.addTests(loader.loadTestsFromTestCase(TestTradeWithRiskManagement))
    test_suite.addTests(loader.loadTestsFromTestCase(TestMoneyIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful()) 