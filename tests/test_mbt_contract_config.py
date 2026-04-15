import os
import sys
import unittest
from decimal import Decimal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants
import money
from config import config


class TestMBTContractConfig(unittest.TestCase):
    def test_contract_constants_match_mbt(self):
        self.assertEqual(constants.CONTRACT_SYMBOL, "MBT")
        self.assertAlmostEqual(constants.CONTRACT_POINT_VALUE, 0.10)
        self.assertAlmostEqual(constants.CONTRACT_TICK_SIZE, 5.0)
        self.assertAlmostEqual(constants.CONTRACT_TICK_VALUE, 0.50)

    def test_money_module_uses_mbt_point_value(self):
        self.assertEqual(money.POINT_VALUE, Decimal("0.10"))
        self.assertEqual(money.TICK_SIZE, Decimal("5.0"))
        self.assertEqual(money.calculate_dollar_change(Decimal("5"), 1), Decimal("0.50"))
        self.assertEqual(money.calculate_dollar_change(Decimal("100"), 10), Decimal("100.0"))

    def test_default_position_size_matches_nq_tick_sensitivity(self):
        self.assertEqual(config["environment"]["position_size"], 10)
        mbt_tick_value_for_default_size = (
            Decimal(str(constants.CONTRACT_TICK_VALUE)) * Decimal(str(config["environment"]["position_size"]))
        )
        self.assertEqual(mbt_tick_value_for_default_size, Decimal("5.0"))


if __name__ == "__main__":
    unittest.main()
