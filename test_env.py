from environment import TradingEnv
import pandas as pd
import numpy as np

def test_position_sizing():
    # Create a simple test DataFrame
    df = pd.DataFrame({
        'Close': [15000, 15020, 15040, 15030, 15050],
        'close_norm': [0.5, 0.51, 0.52, 0.51, 0.53],
        'trend_direction': [1, 1, 1, -1, 1]
    })
    
    # Test with position_size = 1
    env1 = TradingEnv(df, initial_balance=10000, position_size=1)
    obs1, _ = env1.reset()
    print(f"Initial net worth (1 contract): ${env1.net_worth}")
    
    # Take a long position
    action = 0  # Long
    obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
    print(f"Step 1 (1 contract): Action={action}, Position={info1['position']}, Net Worth=${info1['net_worth']}, Position Size={info1['position_size']}")
    
    # Test with position_size = 2
    env2 = TradingEnv(df, initial_balance=10000, position_size=2)
    obs2, _ = env2.reset()
    print(f"Initial net worth (2 contracts): ${env2.net_worth}")
    
    # Take a long position
    action = 0  # Long
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action)
    print(f"Step 1 (2 contracts): Action={action}, Position={info2['position']}, Net Worth=${info2['net_worth']}, Position Size={info2['position_size']}")
    
    # Compare the results
    print("\nComparison:")
    print(f"1 contract change: ${info1['net_worth'] - 10000}")
    print(f"2 contracts change: ${info2['net_worth'] - 10000}")
    print(f"Ratio: {(info2['net_worth'] - 10000) / (info1['net_worth'] - 10000)}")
    
    # Test short position
    print("\nTesting short position:")
    
    # Reset environments
    env1.reset()
    env2.reset()
    
    # Take a short position
    action = 1  # Short
    obs1, reward1, terminated1, truncated1, info1 = env1.step(action)
    obs2, reward2, terminated2, truncated2, info2 = env2.step(action)
    
    print(f"Short position (1 contract): Net Worth=${info1['net_worth']}")
    print(f"Short position (2 contracts): Net Worth=${info2['net_worth']}")
    
    # Compare the results
    print("\nComparison (short):")
    print(f"1 contract change: ${info1['net_worth'] - 10000}")
    print(f"2 contracts change: ${info2['net_worth'] - 10000}")
    print(f"Ratio: {(info2['net_worth'] - 10000) / (info1['net_worth'] - 10000) if (info1['net_worth'] - 10000) != 0 else 'N/A'}")

if __name__ == "__main__":
    test_position_sizing() 