package tests

import w "../wotan/core"
import l "../wotan/linalg"
import ml_finance "../wotan/ml_finance"
import net "../wotan/net"
import nn "../wotan/nn"
import tensor "../wotan/tensor"
import "core:fmt"
import "core:math"
import "core:math/rand"
import "core:mem"

ppo_trading_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== PPO Trading Test (Verbose) ===")

	n_days := 2500
	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	indicators := make([]f64, n_days, allocator)

	price := 100.0
	mean_price := 100.0

	for i in 0 ..< n_days {
		// Ornstein-Uhlenbeck process: pulls price back to mean_price
		reversion_speed := 0.05
		noise := rand.float64_normal(0.0, 1.5)
		price += reversion_speed * (mean_price - price) + noise

		prices[i] = price
		volumes[i] = rand.float64_uniform(1000.0, 10000.0)

		// ✅ Predictive Indicator: Normalized distance from the mean
		// The agent can learn: "If this is high, SELL. If low, BUY."
		indicators[i] = (price - mean_price) / 10.0
	}

	env := ml_finance.new_trading_env(prices, volumes, indicators, 1, 10, 0.001, 10, allocator)
	defer ml_finance.trading_env_free(env)

	obs_dim := env.env.obs_dim
	action_space := env.env.action_space

	agent := ml_finance.new_ppo_agent(
		obs_dim,
		action_space,
		64,
		0.99,
		0.95,
		0.2,
		0.5,
		0.01,
		3e-4,
		allocator,
	)
	defer ml_finance.ppo_agent_free(agent)

	// ✅ Verbose Setup Info
	fmt.println("\n--- Configuration ---")
	fmt.printf(
		"Environment: %d steps | Window: %d | Indicators: %d\n",
		n_days,
		env.window,
		env.n_indicators,
	)
	fmt.printf(
		"Agent:       Obs Dim: %d | Action Space: %d | Hidden Dim: 64\n",
		obs_dim,
		action_space,
	)
	fmt.printf("Buffer Size: %d | Update Threshold: 2048\n", agent.buffer.capacity)
	fmt.println("---------------------\n")

	n_episodes := 10 // ✅ Multiple episodes to see learning
	update_count := 0

	for ep in 0 ..< n_episodes {
		fmt.printf("=== Episode %d/%d ===\n", ep + 1, n_episodes)

		state := ml_finance.env_reset(&env.env)
		ep_reward := 0.0
		done := false
		step_count := 0

		for !done {
			// Create as row vector [1, features] for neural network compatibility
			state_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(state_tensor.data.data, state.data)
			state_tensor.shape = [4]int{1, len(state.data), 1, 1}

			action, log_prob, value := ml_finance.ppo_agent_select_action(agent, state_tensor)
			step := ml_finance.env_step(&env.env, action)

			ml_finance.rollout_buffer_add(
				agent.buffer,
				state_tensor,
				action,
				log_prob,
				step.reward,
				value,
				step.done,
			)

			ep_reward += step.reward
			step_count += 1
			done = step.done

			// ✅ Periodic Step Logging (every 500 steps)
			if step_count % 500 == 0 {
				action_str := "Hold"
				if action == 1 {action_str = "Buy"}
				if action == 2 {action_str = "Sell"}

				fmt.printf(
					"  Step %04d | Action: %-4s | Reward: %+.4f | Inventory: %2d | Cash: $%9.2f | Buffer: %d\n",
					step_count,
					action_str,
					step.reward,
					env.inventory,
					env.cash,
					agent.buffer.size,
				)
			}

			// ✅ PPO Update Logging
			if agent.buffer.size >= 512 {
				update_count += 1
				fmt.printf(
					"  [Update #%d] Buffer full (%d samples). Running PPO update...\n",
					update_count,
					agent.buffer.size,
				)

				stats := ml_finance.ppo_agent_update(agent, 10, 64)

				fmt.printf(
					"    Policy Loss: %.4f | Value Loss: %.4f | Entropy: %.4f | Total Loss: %.4f\n",
					stats.policy_loss,
					stats.value_loss,
					stats.entropy,
					stats.total_loss,
				)
			}

			// ✅ REMOVED: tensor.tensor_free(state_tensor)
			// The buffer takes ownership and will safely free it later in rollout_buffer_clear/free

			state = step.observation
		}

		// ✅ Episode Summary
		fmt.println("\n--- Episode Summary ---")
		fmt.printf("Total Steps:     %d\n", step_count)
		fmt.printf("Final Cash:      $%.2f\n", env.cash)
		fmt.printf("Final Inventory: %d\n", env.inventory)
		fmt.printf("Episode Reward:  %.4f\n", ep_reward)
		fmt.printf(
			"Net PnL:         $%.2f (%.2f%%)\n",
			env.cash - 100000.0,
			(env.cash - 100000.0) / 100000.0 * 100.0,
		)
		fmt.println("=========================\n")
	}

	fmt.printf("Training complete! Total PPO updates: %d\n", update_count)
}
/// ============================================================================
// PPO Trading Test with REAL Market Data (Yahoo Finance)
// ============================================================================
ppo_trading_real_data_test :: proc(allocator: mem.Allocator = context.allocator) {
	fmt.println("\n=== PPO Trading Test (REAL MARKET DATA) ===")

	// ================================================================
	// 1. FETCH REAL DATA
	// ================================================================
	fmt.println("\n[1/5] Fetching real market data from Yahoo Finance...")

	symbol := "SPY"
	spy_df := net.read_yahoo(symbol, .Daily, .FiveYears, allocator)
	defer w.destroy_dataframe(&spy_df)

	if spy_df.rows < 100 {
		fmt.println("ERROR: Failed to fetch enough SPY data. Check network.")
		return
	}
	fmt.printf("  Fetched %d days of %s data\n", spy_df.rows, symbol)

	// ================================================================
	// 2. EXTRACT & PREPROCESS
	// ================================================================
	fmt.println("\n[2/5] Extracting prices, volumes, and computing features...")

	n_days := spy_df.rows
	close_col := w.column(&spy_df, "Close")
	volume_col := w.column(&spy_df, "Volume")

	prices := make([]f64, n_days, allocator)
	volumes := make([]f64, n_days, allocator)
	for i in 0 ..< n_days {
		prices[i], _ = w.column_at_float(close_col, i)
		volumes[i], _ = w.column_at_float(volume_col, i)
	}

	// ----------------------------------------------------------
	// Feature engineering: log returns, RSI, volume z-score
	// ----------------------------------------------------------
	window := 10
	n_indicators := 4 // log_return, rsi, vol_zscore, vol_change

	// Pre-compute log returns
	log_returns := make([]f64, n_days, allocator)
	defer delete(log_returns, allocator)
	for i in 1 ..< n_days {
		if prices[i - 1] > 0 {
			log_returns[i] = math.ln_f64(prices[i] / prices[i - 1])
		}
	}

	// Pre-compute RSI(14)
	rsi_period := 14
	rsi_values := make([]f64, n_days, allocator)
	defer delete(rsi_values, allocator)
	if n_days > rsi_period {
		avg_gain := 0.0
		avg_loss := 0.0
		// Seed initial averages
		for i in 1 ..< rsi_period + 1 {
			change := log_returns[i]
			if change > 0 {
				avg_gain += change
			} else {
				avg_loss += math.abs(change)
			}
		}
		avg_gain /= f64(rsi_period)
		avg_loss /= f64(rsi_period)
		if avg_loss > 1e-10 {
			rs := avg_gain / avg_loss
			rsi_values[rsi_period] = 100.0 - 100.0 / (1.0 + rs)
		} else {
			rsi_values[rsi_period] = 100.0
		}
		// Wilder smoothing
		for i in rsi_period + 1 ..< n_days {
			change := log_returns[i]
			gain := 0.0
			loss := 0.0
			if change > 0 {
				gain = change
			} else {
				loss = math.abs(change)
			}
			avg_gain = (avg_gain * f64(rsi_period - 1) + gain) / f64(rsi_period)
			avg_loss = (avg_loss * f64(rsi_period - 1) + loss) / f64(rsi_period)
			if avg_loss > 1e-10 {
				rs := avg_gain / avg_loss
				rsi_values[i] = 100.0 - 100.0 / (1.0 + rs)
			} else {
				rsi_values[i] = 100.0
			}
		}
	}

	// Pre-compute rolling volume mean and std
	vol_window := 20
	vol_mean := make([]f64, n_days, allocator)
	vol_std := make([]f64, n_days, allocator)
	defer delete(vol_mean, allocator)
	defer delete(vol_std, allocator)
	for i in vol_window ..< n_days {
		sum := 0.0
		for j := i - vol_window; j < i; j += 1 {
			sum += volumes[j]
		}
		vol_mean[i] = sum / f64(vol_window)
		sum_sq := 0.0
		for j := i - vol_window; j < i; j += 1 {
			d := volumes[j] - vol_mean[i]
			sum_sq += d * d
		}
		vol_std[i] = math.sqrt(sum_sq / f64(vol_window))
		if vol_std[i] < 1.0 {
			vol_std[i] = 1.0
		}
	}

	// Build the indicator array that the env expects
	// Layout: n_indicators values per timestep
	indicators := make([]f64, n_days * n_indicators, allocator)
	for i in 0 ..< n_days {
		base := i * n_indicators
		// Feature 0: log return (already small ~±0.03)
		indicators[base + 0] = log_returns[i]
		// Feature 1: RSI normalized to [-1, 1]
		indicators[base + 1] = (rsi_values[i] - 50.0) / 50.0
		// Feature 2: Volume z-score (clamped)
		if i >= vol_window && vol_std[i] > 1.0 {
			z := (volumes[i] - vol_mean[i]) / vol_std[i]
			indicators[base + 2] = math.max(-3.0, math.min(3.0, z)) / 3.0
		}
		// Feature 3: Volume change ratio
		if i >= vol_window && vol_mean[i] > 1.0 {
			indicators[base + 3] =
				math.max(-2.0, math.min(2.0, volumes[i] / vol_mean[i] - 1.0)) / 2.0
		}
	}

	fmt.printf("  Features per step: %d (log_ret, RSI, vol_z, vol_chg)\n", n_indicators)
	fmt.printf("  Observation dim: %d\n", window * (2 + n_indicators))

	// ================================================================
	// 3. CREATE ENVIRONMENT & AGENT
	// ================================================================
	fmt.println("\n[3/5] Creating environment and PPO agent...")

	transaction_fee := 0.001 // 10 bps per trade
	max_position: i32 = 10
	env := ml_finance.new_trading_env(
		prices,
		volumes,
		indicators,
		n_indicators,
		window,
		transaction_fee,
		max_position,
		allocator,
	)
	defer ml_finance.trading_env_free(env)

	obs_dim := env.env.obs_dim
	action_space := env.env.action_space

	agent := ml_finance.new_ppo_agent(
		obs_dim,
		action_space,
		64, // hidden_dim
		0.99, // gamma
		0.95, // gae_lambda
		0.2, // clip_epsilon
		0.5, // value_coef
		0.01, // entropy_coef
		3e-4, // learning_rate
		allocator,
	)
	defer ml_finance.ppo_agent_free(agent)

	fmt.printf("  Obs Dim: %d | Action Space: %d\n", obs_dim, action_space)
	fmt.printf("  Transaction Fee: %.2f%%\n", transaction_fee * 100)

	// ================================================================
	// 4. TRAINING LOOP
	// ================================================================
	fmt.println("\n[4/5] Training...")
	fmt.println("----------------------------------------------------------------------")

	n_episodes := 20
	update_count := 0
	best_pnl := -math.F64_MAX

	// Track cumulative stats across episodes
	ep_pnls := make([dynamic]f64, 0, allocator)
	defer delete(ep_pnls)

	for ep in 0 ..< n_episodes {
		if ep > 0 {
			ml_finance.rollout_buffer_clear(agent.buffer)
		}
		state := ml_finance.env_reset(&env.env)
		ep_reward := 0.0
		done := false
		step_count := 0
		n_trades := 0

		for !done {
			state_tensor := tensor.tensor_new(
				l.matrix_new(f64, 1, len(state.data), allocator),
				false,
				allocator,
			)
			copy(state_tensor.data.data, state.data)
			state_tensor.shape = [4]int{1, len(state.data), 1, 1}

			action, log_prob, value := ml_finance.ppo_agent_select_action(agent, state_tensor)

			prev_inv := env.inventory
			step := ml_finance.env_step(&env.env, action)
			if env.inventory != prev_inv {
				n_trades += 1
			}

			ml_finance.rollout_buffer_add(
				agent.buffer,
				state_tensor,
				action,
				log_prob,
				step.reward,
				value,
				step.done,
			)

			ep_reward += step.reward
			step_count += 1
			done = step.done

			// Log every 500 steps
			if step_count % 500 == 0 {
				action_str := "Hold"
				if action == 1 {action_str = "Buy "}
				if action == 2 {action_str = "Sell"}
				fmt.printf(
					"  Ep %2d | Step %4d | %s | Inv: %+3d | Cash: $%9.2f | Buf: %d\n",
					ep + 1,
					step_count,
					action_str,
					env.inventory,
					env.cash,
					agent.buffer.size,
				)
			}

			// PPO update when buffer is full enough
			if agent.buffer.size >= 512 {
				update_count += 1
				stats := ml_finance.ppo_agent_update(agent, 10, 64)
				if update_count % 5 == 0 {
					fmt.printf(
						"    [Update #%3d] PolLoss: %+.4f | ValLoss: %.4f | Ent: %.4f\n",
						update_count,
						stats.policy_loss,
						stats.value_loss,
						stats.entropy,
					)
				}
			}

			state = step.observation
		}

		pnl := env.cash - 100000.0
		append(&ep_pnls, pnl)
		if pnl > best_pnl {best_pnl = pnl}

		fmt.printf(
			"  Ep %2d/%2d | Steps: %4d | Trades: %3d | Reward: %+.4f | PnL: $%+9.2f (%+.2f%%)\n",
			ep + 1,
			n_episodes,
			step_count,
			n_trades,
			ep_reward,
			pnl,
			pnl / 100000.0 * 100.0,
		)
	}

	// ================================================================
	// 5. FINAL REPORT
	// ================================================================
	fmt.println("\n[5/5] Final Report")
	fmt.println("======================================================================")

	// Cumulative stats
	total_pnl := 0.0
	total_steps := 0
	for p in ep_pnls {total_pnl += p}
	avg_pnl := total_pnl / f64(len(ep_pnls))

	// Simple Sharpe over episode PnLs
	mean_pnl := avg_pnl / 100000.0
	var_pnl := 0.0
	for p in ep_pnls {
		d := p / 100000.0 - mean_pnl
		var_pnl += d * d
	}
	var_pnl /= f64(len(ep_pnls))
	sharpe := 0.0
	if var_pnl > 1e-12 {
		sharpe = mean_pnl / math.sqrt(var_pnl) * math.sqrt(f64(252))
	}

	fmt.printf("  Symbol:            %s\n", symbol)
	fmt.printf("  Episodes:          %d\n", len(ep_pnls))
	fmt.printf("  Total Updates:     %d\n", update_count)
	fmt.printf("  Avg PnL/Episode:   $%+9.2f (%+.2f%%)\n", avg_pnl, avg_pnl / 1000.0)
	fmt.printf("  Best PnL:          $%+9.2f\n", best_pnl)
	fmt.printf("  Sharpe (approx):   %.3f\n", sharpe)
	fmt.println("======================================================================")
}
