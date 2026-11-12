# The `ml_finance` package, part 5: reinforcement learning, order books, text and integration

This is the last part of the guide to `wotan/ml_finance`. The earlier parts:
1. [Part 1](ml_finance_methods_guide.md): labels, feature importance and backtest statistics.
2. [Part 2](ml_finance_asset_pricing_guide.md): asset-pricing models.
3. [Part 3](ml_finance_forecasting_guide.md): forecasting.
4. [Part 4](ml_finance_derivatives_guide.md): derivatives and execution.

This part covers:
- **reinforcement learning:** trading environments, a PPO agent that learns by trading, and offline Conservative Q-Learning (CQL) that learns from a fixed log of trades;
- **order books:** DeepLOB-style classification of limit-order-book windows;
- **graphs:** a correlation graph of assets, and a cross-validated graph attention network;
- **text:** financial sentiment with FinBERT;
- **losses:** a differentiable Sharpe ratio;
- **integration:** the orchestrator, which turns regime, volatility, VRP and HRP signals into allocations, trades and quotes.

Every example was compiled and run against the current code, and the output shown is the real output. A debug build with a tracking allocator reports no leaks for any of them. Section 9 lists the comparisons with CleanRL, PyTorch, NumPy and Hugging Face.

## Contents

1. [Setup and conventions](#1-setup-and-conventions)
2. [Trading environments](#2-trading-environments)
3. [PPO](#3-ppo)
4. [Offline RL: Conservative Q-Learning](#4-offline-rl-conservative-q-learning)
5. [DeepLOB: limit order books](#5-deeplob-limit-order-books)
6. [Graphs of assets and graph attention](#6-graphs-of-assets-and-graph-attention)
7. [Sentiment and the Sharpe loss](#7-sentiment-and-the-sharpe-loss)
8. [The orchestrator](#8-the-orchestrator)
9. [Validation](#9-validation)
10. [Quick reference](#10-quick-reference)

## 1. Setup and conventions

```odin
import ts "wotan/analytics"
import l "wotan/linalg"
import mlf "wotan/ml_finance"
import nn "wotan/nn"
import tok "wotan/nn/tokenizers"
import t "wotan/tensor"
```

As in parts 3 and 4, networks train inside `t.Graph_Scope`s with `nn.Adam`. Two conventions are specific to this part:
- **Observations are owned by the caller.** `env_reset` and `env_step` return an `Observation` whose `data` is allocated with the environment's allocator. Copy it into a tensor and `delete` it.
- **The PPO rollout buffer owns its state tensors.** Pass each state tensor to `rollout_buffer_add` and do not free it; `ppo_agent_update` frees the whole buffer after training on it.

## 2. Trading environments

`Environment` is a small interface: `env_reset(&env)` and `env_step(&env, action)`, with the observation size and the number of actions in `obs_dim` and `action_space`. An episode ends when `current_step` reaches `max_steps`; positions are then closed with the fee. There are two environments.

**`TradingEnv`** (`new_trading_env(prices, volumes, indicators, n_indicators, window, fee, max_position)`) trades one asset:
- **Actions:** 0 hold, 1 buy (10 shares, or 1 if cash is short), 2 sell up to 10. It is long only, with at most one trade per 5 steps and at most `max_position` shares.
- **Observation:** the last `window` days of price and volume relative to the window's first day, plus the indicators.
- **Reward:** shaped. The change in portfolio value over the initial cash, plus 0.1 × the return of a position when it is closed. Minus 2 × the fee for any trade, 1e-5 per share held, and 2 × the drawdown from the peak value.

**`MultiAssetEnv`** (`new_multi_asset_env(prices, volumes, indicators, n_assets, n_indicators, window, action_defs, fee, alloc, risk_aversion = 2, drawdown_coef = 0.1)`) rebalances a portfolio:
- **Data:** prices and volumes are [day, asset], row-major.
- **Actions:** each one is an `ActionDef` of target weights; the remainder is cash.
- **Trading:** a step rebalances at the day's prices, but trades an asset only if its value changes by more than 0.5% of the portfolio, and pays the fee on the traded value.
- **Reward:** mean–variance. With r the portfolio's return for the day (market moves minus fees), the reward is r − `risk_aversion`/2 × r², minus `drawdown_coef` × the *increase* in the drawdown from the peak, plus a small bonus of 1e-4 × cash / initial cash.

The rewards shape behavior, and the shaping matters. Here are four fixed policies on two assets: A drifts up 0.1% a day with 1% volatility, and B has no drift and 2% volatility.

```odin
D, A :: 300, 2 // days, assets: prices [D * A], A drifting up (vol 1%), B flat (vol 2%)
actions := []mlf.ActionDef{{[]f64{0, 0}}, {[]f64{1, 0}}, {[]f64{0, 1}}, {[]f64{0.5, 0.5}}}
env := mlf.new_multi_asset_env(prices, volumes, nil, A, 0, 10, actions) // no indicators, window 10, fee 0.1%
defer mlf.multi_asset_env_free(env)
for a in 0 ..< 4 { // one fixed action for the whole episode
	o := mlf.env_reset(&env.env)
	delete(o.data)
	total := 0.0
	for {
		s := mlf.env_step(&env.env, a)
		delete(s.observation.data)
		total += s.reward
		if s.done {break}
	}
	// env.cash: the final value (everything is sold on the last day)
	// (the decomposition below also sums, per step, the return r, the variance term
	// env.risk_aversion / 2 * r * r and the drawdown increases times env.drawdown_coef;
	// that bookkeeping is left out)
}
```

```
observation 43 values, 4 actions, 290 steps per episode
always all cash : final value 100000.00, summed reward +0.0290 = returns +0.0000 - variance 0.0000 - drawdown 0.0000 + cash bonus 0.0290
always all A    : final value 154469.39, summed reward +0.3320 = returns +0.4489 - variance 0.0281 - drawdown 0.0889 + cash bonus 0.0001
always all B    : final value 130546.79, summed reward +0.0296 = returns +0.3157 - variance 0.0982 - drawdown 0.1881 + cash bonus 0.0001
always half each: final value 143990.83, summed reward +0.2485 = returns +0.3807 - variance 0.0323 - drawdown 0.1001 + cash bonus 0.0001
A 95.73 -> 148.11, B 90.05 -> 117.77 (from day 10, the first trade)
```

The ranking follows return per unit of risk:
- **All in A** earns the best reward, +0.33.
- **Half each** comes second, at +0.25.
- **All in B** made 31% on this path, but its variance and drawdown terms are 3.5 and 2 times A's. That leaves it level with cash.

The two penalties are on the scale of the returns. A day with 1% volatility costs about `risk_aversion`/2 × 1e-4 in variance, against a drift of 1e-3 for A. A fall below the peak is charged once, when it happens, and not again on every day until a new high. Raise `risk_aversion` or `drawdown_coef` to make the agent more cautious.

Check what a reward favors before training on it. If these terms don't suit, write your own environment: `Environment` takes any `reset_fn` and `step_fn`.

**Fixed while writing this guide:** the earlier reward made holding cash the best policy for any realistic asset. Its two penalties were both charged daily:
- **Drawdown:** 0.5 × the drawdown level, so one dip was paid for again each day until a new high.
- **Volatility:** 0.5 × each held asset's mean absolute daily return.

Both scale with the volatility σ, while the gain scales with the drift. Investing would have needed a drift above about 0.4σ a day, an annual Sharpe ratio above 6. Here all in A scored −3.6 and all in B −10.0, against +0.03 for cash.

## 3. PPO

`PPOAgent` is PPO with the clipped surrogate (Schulman et al. 2017), following CleanRL and Stable-Baselines3:
- **Networks:** an actor and a critic, each an MLP with one hidden layer, and one Adam optimizer.
- **Collecting data:** `ppo_agent_select_action` samples an action and returns its log-probability and the critic's value; store each transition with `rollout_buffer_add`.
- **Updating:** `ppo_agent_update(agent, epochs, batch_size, last_value)` computes GAE advantages, normalizes them, and runs `epochs` passes over shuffled minibatches. The loss is the clipped surrogate + `value_coef` × the value error − `entropy_coef` × the entropy. Then it clears the buffer.
- **Rollouts that end mid-episode:** pass `last_value = ppo_agent_value(agent, state)` so the advantages bootstrap from the critic.

`compute_gae` is exposed on its own:

```odin
rewards := []f64{1, 0, 0, 1, 0}
values := []f64{0.5, 0.4, 0.6, 0.7, 0.2}
dones := []bool{false, false, true, false, false} // an episode ends after step 2
adv := mlf.compute_gae(rewards, values, dones, 0.9, 0.8, context.allocator, last_value = 0.3) // gamma, lambda
```

```
advantages [0.6498, -0.2920, -0.6000, 0.5304, 0.0700]
```

The episode boundary cuts the recursion. Step 2's advantage is r − V = −0.6, with no bootstrap.

### Example: a mean-reverting price

The price follows an Ornstein–Uhlenbeck process around 100: 5% reversion a day and noise of 1.5, over 1500 days. The indicator is (price − 100) / 10, and the agent is updated every 512 transitions. The example runs 60 episodes, about 8 s.

```odin
env := mlf.new_trading_env(prices, volumes, ind, 1, 10) // 1 indicator, window 10, fee 0.1%, at most 10 shares
defer mlf.trading_env_free(env)
agent := mlf.new_ppo_agent(env.env.obs_dim, env.env.action_space) // hidden 64, gamma 0.99, lambda 0.95, clip 0.2
defer mlf.ppo_agent_free(agent)

obs_tensor :: proc(o: mlf.Observation) -> ^t.Tensor { // copies and frees the observation
	x := t.tensor_new(l.matrix_new(f64, 1, len(o.data)))
	copy(x.data.data, o.data)
	delete(o.data)
	return x
}
for ep in 0 ..< 60 {
	state := obs_tensor(mlf.env_reset(&env.env))
	for {
		action, log_prob, value := mlf.ppo_agent_select_action(agent, state)
		step := mlf.env_step(&env.env, action)
		mlf.rollout_buffer_add(agent.buffer, state, action, log_prob, step.reward, value, step.done) // owns state
		state = obs_tensor(step.observation)
		if agent.buffer.size >= 512 {
			last := mlf.ppo_agent_update(agent, 10, 64, last_value = step.done ? 0 : mlf.ppo_agent_value(agent, state))
		}
		if step.done {break}
	}
	t.tensor_free(state) // the state after the last step is not in the buffer
	// env.cash - 100000: the episode's P&L
}
```

```
episode 10: P&L   +140.34, 136 trades, summed reward -3.812; last update: policy -0.0117 value 0.00252 entropy 0.773
episode 20: P&L   +185.88,  94 trades, summed reward -2.981; last update: policy -0.0121 value 0.00089 entropy 0.456
episode 30: P&L   +174.18,  70 trades, summed reward -2.932; last update: policy -0.0071 value 0.00035 entropy 0.353
episode 40: P&L   +351.96,  58 trades, summed reward -2.277; last update: policy -0.0063 value 0.00021 entropy 0.278
episode 50: P&L    +49.89,  30 trades, summed reward -2.270; last update: policy -0.0060 value 0.00028 entropy 0.193
episode 60: P&L   +116.95,  42 trades, summed reward -2.122; last update: policy -0.0061 value 0.00016 entropy 0.188
```

The agent learns what the reward asks for:
- **Its entropy** falls from near the uniform ln 3 = 1.10 to 0.19, so the policy becomes nearly deterministic.
- **It trades less:** about 210 trades in the first episode, about 40 by the end.
- **The summed reward** rises from about −11 to −2, mostly through fewer trading and drawdown penalties.

The P&L stays small: positions are capped at 10 shares, about $1,000 on $100,000. A few hundred dollars per episode is a modest profit from the mean reversion. In the first episodes, a near-random policy lost up to $570.

RL in trading is fragile. Judge an agent on held-out data, as `walk_forward_validation_test` does, not by its training curve.

## 4. Offline RL: Conservative Q-Learning

Online RL learns by trading. Offline RL learns from a log of past decisions, such as a desk's trades or a rule's backtest. Plain Q-learning fails there: it over-values actions the log rarely took, and the greedy policy then exploits those errors.

CQL (Kumar et al. 2020, CQL(H)) adds a regularizer, `alpha` × mean(logsumexp_a Q(s, a) − Q(s, a_logged)). It pushes down the values of actions the log did not take:

  loss = mean (Q(s, a) − y)² + α · mean(logsumexp_a Q(s, ·) − Q(s, a)), with y = r + γ (1 − done) max_a Q_target(s′, a).

The CQL tools:
- **The network:** `cql_network_new`, a one-hidden-layer MLP. A target copy follows it through `cql_soft_update`.
- **The loss:** `cql_loss` returns the total, the TD error and the regularizer.
- **Data:** `cql_trading_dataset(closes, actions, cost)` turns a log of short / flat / long decisions into transitions. A state holds the last 3 returns in percent and the position held into the day; the reward is the next day's P&L in percent of notional, net of costs.
- **Synthetic data:** `generate_trading_dataset` makes such a log from AR(1) returns (φ = 0.3, so there is momentum to earn), traded by a noisy momentum rule.
- **Evaluation:** `cql_backtest` runs the greedy policy over a price path.

```odin
config := mlf.CQLConfig{state_dim = mlf.CQL_STATE_DIM, num_actions = 3, hidden_dim = 32, gamma = 0.9, alpha = 1.0, tau = 0.01}
states, actions, rewards, next_states, dones, closes := mlf.generate_trading_dataset(1000, context.allocator)
q := mlf.cql_network_new(config)
defer mlf.cql_network_free(&q)
target := mlf.cql_network_new(config)
defer mlf.cql_network_free(&target)
mlf.cql_network_copy(&q, &target)
opt := nn.adam_new(3e-3)
defer nn.adam_free(&opt)
mlf.cql_network_add_to_adam(&q, &opt)
for step in 0 ..< 1500 { // full batch
	t.graph_begin(&scope)
	nn.adam_zero_grad(&opt)
	qv := mlf.cql_network_forward(&q, states)
	nq := mlf.cql_network_forward(&target, next_states)
	loss, td, reg := mlf.cql_loss(qv, actions, rewards, nq, dones, config, context.allocator)
	t.tensor_backward(loss)
	nn.adam_step(&opt)
	t.graph_end(&scope)
	mlf.cql_soft_update(&q, &target, config.tau)
}
pnl, counts := mlf.cql_backtest(&q, closes, 0.001) // summed daily P&L in %, decisions per action
```

```
first state [-0.678, -0.056, 0.799, 0.000], action 1, reward -0.0000
step    0 loss 2.2694 (TD 1.0540, regulariser 1.2155)
step  500 loss 1.2849 (TD 0.6904, regulariser 0.5944)
step 1000 loss 1.2432 (TD 0.6640, regulariser 0.5792)
step 1499 loss 1.2147 (TD 0.6503, regulariser 0.5644)
in sample: CQL +118.9% (short/flat/long [313, 347, 340]), logged rule +70.1% ([325, 345, 330])
fresh path: CQL +166.9% (short/flat/long [322, 332, 346]), logged rule +126.1%
alpha 0 (plain Q-learning): fresh path -154.7% ([327, 237, 436])
```

How to read the results:
- **CQL improves on the rule that produced its data.** In sample it makes +119% against the rule's +70%. On a fresh path of the same process it makes +167% against +126%.
- **It stays close to the logged behavior,** taking each action about as often as the rule did.
- **The same network trained with α = 0, plain Q-learning on the log,** loses 155% on the fresh path. It drifts toward going long, which the log does not support. This is the failure CQL exists to prevent.

## 5. DeepLOB: limit order books

`DeepLOB` follows Zhang, Zohren & Roberts (2019), in simplified form:
- **Input:** windows [batch, 3, T, levels] holding the mid, the spread and the volume imbalance per level.
- **Network:** two 3×3 convolutions, 2×2 max pooling, an inception block (1×1, 3×3 and 5×5 branches), then a sequence layer (LSTM, GRU or Mamba) and a linear head.
- **Output:** the logits of down, stationary and up.

Turning a book into windows takes two steps:
1. **`deeplob_lobster_windows(book, n_rows, levels, T, horizon, threshold)`** cuts an order book in LOBSTER layout (per level: ask price, ask size, bid price, bid size) into windows of T events. Each window is labeled with the move of the mean mid over the next `horizon` events, against `threshold`. It also returns the realized future return.
2. **`deeplob_norm_fit`** fits a z-score per channel and level on the training windows only; apply it to all windows with `deeplob_norm_apply`.

The example is a synthetic 4-level book of 6000 events. The level-1 imbalance follows an AR(1) and tilts the next mid move: a buy-heavy book tends to tick up.

```odin
x, labels, fut := mlf.deeplob_lobster_windows(book, 6000, 4, 20, 10, 2e-5) // T 20, horizon 10
defer {t.tensor_free(x); delete(labels); delete(fut)}
// training: the first 70% of the windows; test: from 30 events after them (no shared event)
nm := mlf.deeplob_norm_fit(x_train)
defer mlf.deeplob_norm_free(&nm)
mlf.deeplob_norm_apply(x_train, nm)
mlf.deeplob_norm_apply(x_test, nm)
model := mlf.deeplob_new({time_steps = 20, price_levels = 4, num_classes = 3, hidden_dim = 16, seq_backend = .LSTM})
defer mlf.deeplob_free(model)
opt := nn.adam_new(1e-3)
defer nn.adam_free(&opt)
mlf.deeplob_add_to_adam(model, &opt)
// per batch of 64, in a graph scope:
//   loss := t.tensor_cross_entropy_loss(mlf.deeplob_forward(model, xb), yb)
// evaluation: mlf.deeplob_eval(model) turns the dropout off (deeplob_train turns it back on)
```

```
5971 windows of shape [5971, 3, 20, 4]
window 0, last event: mid 99.980, spread 0.010, imbalance by level [-0.900, 0.278, 0.357, -0.046]
test labels down/flat/up [819, 133, 810]: always '0' scores 46.5%
epoch 1: train loss 0.9358, test accuracy 61.4%
epoch 2: train loss 0.8642, test accuracy 61.7%
epoch 3: train loss 0.8461, test accuracy 61.7%
epoch 4: train loss 0.8341, test accuracy 62.3%
```

Four epochs (8 s) give 62% held-out accuracy against 46.5% for always predicting the most frequent class. The network found the imbalance signal. `compute_lob_microstructure` builds the three channels from raw bid and ask prices and sizes, if your data are not in LOBSTER layout.

## 6. Graphs of assets and graph attention

`compute_correlation_adjacency(features, n_assets, time_steps)` builds an asset graph from features [windows, assets, time, 1]:
- **Correlation:** the Pearson correlation of each pair over all windows.
- **Edges:** negative correlations are set to 0, and every asset gets a self-loop.
- **Normalization:** each row is scaled to sum 1.

`gat_cross_val_score(features, targets, n_splits, hidden, heads, epochs, lr)` cross-validates a graph attention layer (`nn.gat_layer_new`) with a linear head that predicts each asset's next return:
- **Folds** are contiguous blocks, purged as in [part 1](ml_finance_methods_guide.md): training windows near the validation block are dropped.
- **The adjacency** comes from the training windows only.
- **Training** is full-batch Adam, with early stopping on the validation MSE (patience 10), keeping the best epoch.
- **Each fold** reports the cross-sectional rank IC of the validation predictions, from `cross_sectional_rank_ic` in part 1. The procedure also prints one line per fold.

The example has 120 windows of 5 days for 6 assets in two sectors of 3. An asset's next return is 0.3 × its sector's last move + noise.

```odin
feat.shape = {120, 6, 5, 1} // returns
targ.shape = {120, 6, 1, 1} // the next return
adj := mlf.compute_correlation_adjacency(feat, 6, 5, context.allocator)
res := mlf.gat_cross_val_score(feat, targ, 4, 8, 2, 100, 0.01) // 4 folds, hidden 8, 2 heads, <= 100 epochs
defer delete(res)
```

```
adjacency row 0 [0.382, 0.312, 0.302, 0.004, 0.000, 0.000]
adjacency row 3 [0.003, 0.012, 0.006, 0.372, 0.301, 0.305]
  Fold 1/4: Mean IC = +0.1981, Val Loss = 0.9136 (stopped at epoch 39)
  Fold 2/4: Mean IC = +0.0362, Val Loss = 0.9907 (stopped at epoch 17)
  Fold 3/4: Mean IC = +0.0248, Val Loss = 1.1406 (stopped at epoch 16)
  Fold 4/4: Mean IC = -0.0095, Val Loss = 1.0266 (stopped at epoch 21)
```

The graph separates the two sectors cleanly. The predictive signal is weak: the noise has more than three times the signal's standard deviation, there are only 90 training windows per fold, and the IC is measured across just 6 assets. The mean IC of 0.06, positive in 3 of 4 folds, is the honest result. Variation of this size across folds is why a single train/test split misleads.

## 7. Sentiment and the Sharpe loss

### FinBERT

`SentimentAnalyzer` is BERT with a 3-class head, as Hugging Face's `BertForSequenceClassification`:
- **`sentiment_analyzer_load(path)`** reads a FinBERT checkpoint (ProsusAI/finbert). To make one, convert it with `scripts/convert_bert.py`, as the [test catalog](test_catalog.md) describes.
- **`sentiment_analyzer_new`** makes an untrained model to fine-tune.
- **Labels.** The order of the classes differs between FinBERT and a new model. Read the results through `sentiment_probabilities` or `sentiment_predict`, which map them to `SentimentLabel`s, rather than by position.

```odin
analyzer, ok := mlf.sentiment_analyzer_load("model_checkpoints/finbert_checkpoint.bin")
defer mlf.sentiment_analyzer_free(&analyzer)
tk, tok_ok := tok.wordpiece_tokenizer_new("model_checkpoints/bert_vocab.txt", 48) // at most 48 tokens
defer tok.wordpiece_tokenizer_free(&tk)
for s in sentences {
	ids, seg := tok.tokenize_to_tensors(&tk, s)
	t.graph_begin(&scope, no_grad = true)
	logits := mlf.analyze_text(&analyzer, ids, seg, false)
	probs := mlf.sentiment_probabilities(&analyzer, logits) // [SentimentLabel]f64
	label, p := mlf.sentiment_predict(&analyzer, logits)
	t.graph_end(&scope)
	t.tensor_free(ids)
	t.tensor_free(seg)
}
```

```
The company reported record quarterly revenue and raised its full-year guidance.   Positive 0.957  (neg 0.021 neu 0.023 pos 0.957)
Shares plunged after the firm missed earnings estimates and cut its dividend.      Negative 0.970  (neg 0.970 neu 0.021 pos 0.009)
The board will meet on Tuesday to discuss the annual report.                       Neutral  0.932  (neg 0.045 neu 0.932 pos 0.023)
Margins came under pressure, although sales grew slightly.                         Positive 0.581  (neg 0.388 neu 0.031 pos 0.581)
```

The clear cases are clear. The mixed sentence is split between positive and negative, and the low probability (0.58) says so. Use the probabilities, not only the label. The [event study in part 2](ml_finance_asset_pricing_guide.md#5-event-studies) turns such scores into event dates.

### The Sharpe loss

`sharpe_loss(returns, rf)` is −(mean − rf) / √(var + 1e-8) over all entries of a returns tensor, with the population variance. It is differentiable, so a model can be trained to maximize the Sharpe ratio of the returns its positions produce, as `deeplob_real_data_sharpe_test` does.

```odin
r := t.tensor_new(l.matrix_new(f64, 1, 4), true)
copy(r.data.data, []f64{0.01, -0.005, 0.02, 0.0})
t.graph_begin(&scope)
loss := mlf.sharpe_loss(r)
t.tensor_backward(loss)
t.graph_end(&scope)
```

```
loss -0.650909 (minus mean / population sd), gradient [-19.4177, -45.8925, -1.7678, -37.0675]
```

The gradient matches central finite differences. The loss pools every entry, so it is one Sharpe ratio for the whole tensor, not one per row. It is not annualized.

## 8. The orchestrator

The orchestrator turns the signals of the earlier parts into a portfolio decision. The caller computes the inputs; `orchestrator_step` applies the rules:
- **Action.** `regime_to_action` maps the current regime's annualized mean and volatility to an action:
  - volatility above 30% with a negative mean: Liquidate;
  - volatility above 25%: Reduce;
  - mean above 10% with volatility below 15%: Aggressive;
  - otherwise: Hold.
- **Allocation.** Each action has an equity share: 85% (Aggressive), 65% (Hold), 40% (Reduce), 10% (Liquidate). The share is split over the equities by their HRP weights; the rest goes to the bond, which is by convention the last asset.
- **Exposure.** A multiplier scales the whole allocation, and the rest is cash. It is the action's base (1.5, 1, 0.5 or 0) × (1 + 0.3 × the VRP signal) × the confidence factor 0.5 + 0.5 × confidence, where the confidence comes from the width of the volatility interval. The multiplier is capped at `max_gross` (default 1: no leverage).
- **Trades.** The shares to trade are (target − current weight) × capital / price, assumed filled at that price.
- **Quotes.** The Avellaneda–Stoikov quotes of [part 4](ml_finance_derivatives_guide.md#2-optimal-execution-and-market-making) are computed around the price, with the shares held as inventory.

`orchestrator_mark_to_market(&state, returns, capital)` then applies the next period's returns. It adds the P&L and lets the weights drift with prices.

### Example: a regime-switching market

The market has 750 days of two equities and a bond, switching between two regimes:
- calm: equity volatility 12%, drift +12% a year;
- stressed: equity volatility 35%, drift −40% a year, and the bond gains a little.

The last weeks are calm until day 735, then stressed. The pipeline, with each input using only data up to the decision day:
- **Regime:** an HMM fitted on the equal-weighted equity return of the first 500 days; the regime on day d is the filtered state.
- **Volatility:** GARCH(1,1) gives the volatility forecast, with an interval of ±20%.
- **VRP signal:** `compute_vrp` with a VIX of 20.
- **Weights:** HRP on the trailing 60-day covariance.

The orchestrator runs from day 700 to 749, and is compared with fixed equal weights.

```odin
h := mlf.hmm_new(2)
defer mlf.hmm_free(&h)
mlf.hmm_init_simple(&h, eq[:500])
mlf.hmm_fit(&h, eq[:500])
g := ts.garch_fit_spec(eq[:500], {p = 1, q = 1})
defer ts.garch_result_free(&g)
gv := mlf.garch_recursive_forecast(eq, g.params.omega, g.params.alpha[0], g.params.beta[0])
defer delete(gv)
state := mlf.orchestrator_init(3) // starts at equal weights
defer mlf.orchestrator_free(&state)
as := mlf.AvellanedaStoikovParams{sigma = 1, gamma = 1e-4, k = 10, T = 1}
for d in 700 ..< 749 {
	filt := mlf.hmm_filter(&h, eq[:d + 1])
	regime := 1 if filt[d * 2 + 1] > 0.5 else 0
	delete(filt)
	vol := gv[d + 1] * math.sqrt(252.0)
	// cov: the trailing 60-day covariance of the three assets
	hrp := mlf.hrp_allocate_cov(&cov)
	vrp := mlf.compute_vrp(20, gv[d + 1], 3, 3)
	dec := mlf.orchestrator_step(&state, regime, 252 * h.emission_mu[regime], math.sqrt(252 * h.emission_var[regime]),
		vol, 0.8 * vol, 1.2 * vol, vrp.signal, hrp.weights, prices, capital, as)
	if d == 700 || d == 736 {mlf.print_decision(&dec, tickers, state.day)}
	mlf.decision_free(&dec, state.allocator)
	mlf.hrp_result_free(&hrp)
	rp := mlf.orchestrator_mark_to_market(&state, next_returns, capital * wealth)
	wealth *= 1 + rp
}
```

```
regimes: annualised mean 0.163 / -1.323, vol 0.106 / 0.297

╔══════════════════════════════════════════════════════════════╗
║  ORCHESTRATOR DECISION, day 1                                ║
╠══════════════════════════════════════════════════════════════╣
║  Regime: 1 | Action: REDUCE     | Pos. mult: 0.32            ║
║  Vol forecast: 34.86% | CI: [27.89%, 41.83%]                 ║
║  VRP signal: -0.999 | Exposure:  32% (rest cash)             ║
╠══════════════════════════════════════════════════════════════╣
║  Asset  | Target wt | Trade (sh) |      Bid |      Ask       ║
╠══════════════════════════════════════════════════════════════╣
║  EQ1    |      6.6% |      -2669 |    99.87 |   100.07       ║
║  EQ2    |      6.0% |      -2738 |    99.87 |   100.07       ║
║  BOND   |     18.9% |      -1443 |    99.81 |   100.01       ║
╚══════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════╗
║  ORCHESTRATOR DECISION, day 37                               ║
╠══════════════════════════════════════════════════════════════╣
║  Regime: 0 | Action: AGGRESSIVE | Pos. mult: 1.63            ║
║  Vol forecast: 13.18% | CI: [10.55%, 15.82%]                 ║
║  VRP signal: +0.690 | Exposure: 100% (rest cash)             ║
╠══════════════════════════════════════════════════════════════╣
║  Asset  | Target wt | Trade (sh) |      Bid |      Ask       ║
╠══════════════════════════════════════════════════════════════╣
║  EQ1    |     41.1% |        -58 |   105.05 |   105.25       ║
║  EQ2    |     43.9% |        +51 |   105.44 |   105.64       ║
║  BOND   |     15.0% |         +6 |    99.31 |    99.51       ║
╚══════════════════════════════════════════════════════════════╝
days 701-749: orchestrator +5.82% (P&L 58160), equal weights +3.94%
regimes used: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
```

How to read the two decisions:
- **Day 1** falls in a stressed spell. The regime's 30% volatility gives Reduce. The volatility forecast of 35%, above the VIX of 20, makes the VRP signal −1. The exposure drops to 32%, and the trades sell down from the starting equal weights.
- **Day 37** is calm. Aggressive with a positive VRP signal gives a multiplier of 1.63, capped at 100% exposure, with 85% in equities split by HRP. The trades are small because the weights only drift between days.

The regime list for days 31–49 shows the filtered HMM switching six days after the stress begins on day 36, and holding from day 44. That lag is the price of not looking ahead.

The orchestrator beat equal weights over these 49 days. That is one path of a market built to reward regime timing; it is not evidence. `orchestrator_test` walks the orchestrator forward over a year of real data, against 60/40, equal weights and SPY. It reports each strategy's PSR, its minimum track record length, and its costs.

**Fixed while writing this guide:** `print_decision`'s right border was misaligned on most rows. Every row is now padded to the box.

## 9. Validation

The tests are registered in `main.odin` and listed in [the test catalog](test_catalog.md):

| Area | Reference | Test |
|---|---|---|
| PPO and the environments | GAE vs CleanRL (NumPy); the minibatch loss gradient vs finite differences of the clipped objective; buffer growth; both environments on scripted episodes, the multi-asset rewards vs an independent Python replay | `ppo_reference_test` |
| CQL | the CQL(H) loss and dL/dQ vs PyTorch; the network gradient vs finite differences; the trading dataset vs NumPy; a one-state bandit trained to SciPy's minimizer | `cql_reference_test` |
| DeepLOB | windows, labels, future returns and the normalizer vs NumPy on a LOBSTER book; gradients vs finite differences for every sequence backend | `deeplob_reference_test`, `deeplob_test` |
| Graphs | the correlation adjacency vs `np.corrcoef`; GAT cross-validation on a tracking allocator; the GAT layer vs PyTorch | `ml_finance_metrics_reference_test`, `models_reference_test` |
| FinBERT | WordPiece ids, logits and labels of four sentences vs Hugging Face (logits to 1e-12) | `sentiment_reference_test` |
| Orchestrator | every rule vs hand-computed values; `hmm_filter` vs NumPy | `orchestrator_reference_test` |

The application tests:
- **PPO:** `ppo_test` on synthetic and real data, with walk-forward validation; `ppo_multi_asset_test` on SPY and QQQ.
- **CQL:** `cql_trading_test` and `cql_real_data_test`, on AAPL.
- **DeepLOB:** `deeplob_real_data_test`, on the LOBSTER TSLA sample.
- **GAT:** `gat_cross_sectional_test`.
- **Orchestrator:** `orchestrator_test`.

For this guide, the GAE values were also recomputed in Python (CleanRL's recursion) and the Sharpe-loss gradient by central differences; both agree to the printed decimals.

## 10. Quick reference

### Environments and PPO

| Procedure | Purpose |
|---|---|
| `new_trading_env(prices, volumes, indicators, n_ind, window, fee, max_position)` / `trading_env_free` | one-asset environment |
| `new_multi_asset_env(prices, volumes, indicators, n_assets, n_ind, window, action_defs, fee, alloc, risk_aversion, drawdown_coef)` / `multi_asset_env_free` | portfolio environment, mean–variance reward |
| `env_reset(&env)` / `env_step(&env, action)` | episode (observations: caller deletes `data`) |
| `new_ppo_agent(obs_dim, actions, hidden, gamma, lambda, clip, value_coef, entropy_coef, lr)` / `ppo_agent_free` | the agent |
| `ppo_agent_select_action(agent, state)` / `ppo_agent_value(agent, state)` | action, log-probability, value |
| `rollout_buffer_add(buffer, state, action, log_prob, reward, value, done)` | store a transition (owns the state) |
| `ppo_agent_update(agent, epochs, batch_size, last_value)` | PPO update; `PPOUpdateStats` |
| `compute_gae(rewards, values, dones, gamma, lambda, alloc, last_value)` | advantages |
| `ppo_minibatch_loss` | the loss of one minibatch (graph) |

### CQL

| Procedure | Purpose |
|---|---|
| `cql_network_new(CQLConfig{...})` / `cql_network_free` / `cql_network_add_to_adam` | Q-network |
| `cql_network_forward` / `cql_greedy_action` | Q-values, greedy action |
| `cql_network_copy` / `cql_soft_update(main, target, tau)` | target network |
| `cql_loss(q, actions, rewards, next_q, dones, config, alloc)` | loss, TD error, regularizer |
| `cql_trading_dataset(closes, actions, cost)` / `cql_trading_state` | transitions from a trading log |
| `generate_trading_dataset(n, alloc, phi)` / `generate_ar1_closes` | synthetic log |
| `cql_backtest(&net, closes, cost)` | greedy policy P&L |

### Order books, graphs, text, losses

| Procedure | Purpose |
|---|---|
| `deeplob_new(DeepLOBConfig{...})` / `deeplob_free` / `deeplob_add_to_adam` | the model |
| `deeplob_forward(m, x)` / `deeplob_train` / `deeplob_eval` | logits, dropout on / off |
| `deeplob_lobster_windows(book, n_rows, levels, T, horizon, threshold)` | windows, labels, future returns |
| `compute_lob_microstructure(raw, n, T, levels, alloc)` | mid / spread / imbalance channels |
| `deeplob_norm_fit` / `deeplob_norm_apply` / `deeplob_norm_free` | train-set normalization |
| `compute_correlation_adjacency(features, n_assets, T, alloc)` | asset graph |
| `gat_cross_val_score(features, targets, splits, hidden, heads, epochs, lr)` | purged CV of a GAT; `GATFoldResult` |
| `sentiment_analyzer_load(path)` / `sentiment_analyzer_new(...)` / `sentiment_analyzer_free` | FinBERT / a new classifier |
| `analyze_text(&a, ids, segments, training)` | logits |
| `sentiment_probabilities` / `sentiment_predict` | by label |
| `sharpe_loss(returns, rf)` | differentiable −Sharpe |

### Orchestrator

| Procedure | Purpose |
|---|---|
| `orchestrator_init(n_assets)` / `orchestrator_free` | state (equal weights) |
| `orchestrator_step(&state, regime, mean, vol, vol_forecast, lo, hi, vrp, hrp_weights, prices, capital, as_params, max_gross)` | one decision; `decision_free` |
| `orchestrator_mark_to_market(&state, returns, capital)` | P&L and weight drift |
| `regime_to_action` / `compute_position_multiplier` / `orchestrator_equity_fraction` / `orchestrator_confidence` | the rules |
| `compute_rebalance_trades` / `generate_quotes` | trades and quotes |
| `print_decision(&d, tickers, day)` | the dashboard |
