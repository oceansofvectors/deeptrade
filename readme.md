Below are the areas I would tackle first, roughly in the order I expect them to move the P & L needle the most.  I highlight **what to change, why it matters for a 5‑minute NQ scalper, and exactly where in your code/config you’d touch it**.

---

### 1  Data depth & internal feature set  
* **Train on at least a year of bars** instead of the 60‑day slice in `config.yaml` (`period: "60d"`) citeturn0file0.  PPO needs tens‑of‑millions of state–action pairs before it begins to generalise; with NQ you can easily pull 12‑18 months of 5‑minute data from CME or Polygon.  
* **Turn on order‑flow & micro‑structural signals** – e.g. VWAP, bid/ask imbalance, rolling volume delta.  Your env already accepts arbitrary columns; just enable them in the `indicators:` block and add the calculations in `get_data.py`.  
* **Encode session context instead of `minutes_since_open` alone**.  Replace the scalar with two cyclical features (`sin`/`cos`) of *minutes since RTH open* and another pair for *minutes until close*.  It gives the network a smoother representation of where it is in the session.

---

### 2  Reward shaping & position representation  
* Current reward is the *log‑return of equity each step* citeturn1file18.  That’s neat academically, but for a scalper it ties reward to *holding time* as much as to directional accuracy.  
  * **Switch to instantaneous tick P & L (after costs) per bar**, scaled by initial capital.  
  * Add **penalties for cheap flips** (e.g. −0.2 every time the action toggles) to discourage churn.  
* Action space is discrete {long, short, flat}.  Consider a **parametric action head** where the first logit decides *direction* and a second, continuous value decides *fraction of max size* – it lets the policy size in or out gradually without exploding the observation space.  You can prototype it quickly with SB3’s `MlpPolicy` + `action_net` override.

---

### 3  Transaction‑cost realism  
* `transaction_cost` is **0** in both the env and config citeturn0file0turn1file16. Live, every NQ round‑turn is ≈\$2.40 + slippage.  Set `transaction_cost` to 0.00012 (≈1.2 bp on current price) **and** deduct a constant \$2.40 per filled order in `RiskManager.exit_position()` citeturn1file17.  
* Slippage: sample ±0–1 tick around the close price each time a trade is executed; that alone kills many over‑fitted high‑frequency strategies.

---

### 4  Tighter, asymmetric risk management  
* Take‑profit is 20 %(!) of equity while stop‑loss is disabled citeturn0file0 – the inverse of what a scalper needs.  
  * Enable `stop_loss` at 0.15–0.25 % of *entry equity*.  
  * Move `take_profit` to 0.30–0.50 % or just use a **trailing stop** of ~0.35 % (`trailing_stop.enabled true`).  
  * Feed the **distance to stop/TP** into the observation vector so the agent *knows* how close it is to being forced out.  
* Add a **daily loss circuit‑breaker** (max 1 % of start‑of‑day equity).  You already implemented the plumbing in `RiskManager.check_daily_risk_limit()` citeturn1file4 – just flip the `daily_risk_limit.enabled` switch to `true`.

---

### 5  Model architecture & training loop  
* **Vectorise the environment** (`SubprocVecEnv` or `DummyVecEnv`) so PPO sees multiple independent market days in parallel; that speeds learning and makes the value‑function less over‑confident on one path.  
* Swap the default MLP for **Temporal Convolution (TCN) or a small GRU head** – 5‑minute series have strong short‑term autocorrelation that an MLP can’t capture.  SB3’s `RecurrentPPO` will drop‑in here.  
* Use **learning‑rate decay** (cosine or linear) over each walk‑forward window; right now LR is fixed at ≈3 e‑3 citeturn0file0 which is high for PPO on noisy returns – start at 1 e‑3 and decay to 1 e‑4.  


---

### 7  Code & performance polish  
* **Duplicate P & L logic** exists in both the env and `RiskManager`; ensure the same helper (e.g. `money.calculate_price_change`) drives both to avoid subtle drift.  
* Use `np.asarray` / `df.values` when building the observation to cut Pandas overhead; it matters when you upgrade to thousands of episodes per epoch.  
* Add a **unit test that replays one historical day** and asserts the env’s cumulative P & L equals the `RiskManager`’s – that single test will catch 90 % of future bugs.  

---

### 8  Next steps to validate  
1. **Replay** ten random recent days through the trained policy with the new cost/slippage model.  You should still see a hit‑rate >50 % and average net P & L per contract >\$5 per trade (after fees).  
2. **Shadow‑trade in SIM** for two weeks, saving tick‑level fills so you can verify reward‑to‑live drift.  
3. If live drift is acceptable, start with 1 micro‑NQ contract and scale only when the Sharpe on live fills matches back‑test ±0.2.  

Implementing even half the items above usually halves back‑test Sharpe but *doubles* the likelihood the strategy survives first contact with real ticks.