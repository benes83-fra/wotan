// wotan/finance/backtest.odin
package finance

import w "../core"
import l "../linalg"
import p "../plot"
import "core:math"
import "core:mem"

// ============================================================================
// Configuration
// ============================================================================

BacktestConfig :: struct {
	initial_capital: f64,
	commission_rate: f64,
	slippage_rate:   f64,
	risk_free_rate:  f64,
}

DEFAULT_BACKTEST_CONFIG :: BacktestConfig {
	initial_capital = 100000.0,
	commission_rate = 0.001,
	slippage_rate   = 0.0005,
	risk_free_rate  = 0.02,
}

// ============================================================================
// Order Management
// ============================================================================

OrderType :: enum {
	Market,
	Limit,
	Stop,
}

OrderSide :: enum {
	Buy,
	Sell,
}

Order :: struct {
	symbol:      string,
	side:        OrderSide,
	quantity:    f64,
	order_type:  OrderType,
	limit_price: f64,
	stop_price:  f64,
}

FilledOrder :: struct {
	order:      Order,
	fill_price: f64,
	fill_time:  int,
	commission: f64,
}

// ============================================================================
// Position Tracking
// ============================================================================

// quantity is signed (< 0 = short); avg_price is the average entry price of the
// open quantity. A round trip runs from flat back to flat (a fill that flips
// the position closes one trip and opens the next).
Position :: struct {
	symbol:         string,
	quantity:       f64,
	avg_price:      f64,
	unrealized_pnl: f64,
	realized_pnl:   f64, // price P&L of closed quantity, before commissions
	round_trips:    int,
	winning_trades: int, // round trips with positive P&L net of commissions
	is_short:       bool,
	trade_pnl:      f64, // P&L of the open round trip so far, net of its commissions
}

// ============================================================================
// Portfolio State
// ============================================================================

Portfolio :: struct {
	cash:         f64,
	positions:    map[string]Position,
	equity_curve: [dynamic]f64,
	trades:       [dynamic]FilledOrder,
	current_bar:  int,
	trade_pnls:   [dynamic]f64, // net P&L of each closed round trip
}

portfolio_init :: proc(capital: f64, allocator: mem.Allocator) -> Portfolio {
	return Portfolio {
		cash = capital,
		positions = make(map[string]Position, allocator),
		equity_curve = make([dynamic]f64, 0, allocator),
		trades = make([dynamic]FilledOrder, 0, allocator),
		current_bar = 0,
		trade_pnls = make([dynamic]f64, 0, allocator),
	}
}

@(private = "file")
_portfolio_destroy :: proc(p: ^Portfolio) {
	delete(p.positions)
	delete(p.equity_curve)
	delete(p.trades)
	delete(p.trade_pnls)
}

// Cash plus signed position values (positions without a price count as 0)
portfolio_total_value :: proc(p: ^Portfolio, prices: map[string]f64) -> f64 {
	total := p.cash
	for symbol, pos in p.positions {
		if price, ok := prices[symbol]; ok {
			total += pos.quantity * price
		}
	}
	return total
}

portfolio_update :: proc(p: ^Portfolio, prices: map[string]f64) {
	for symbol, &pos in p.positions {
		if price, ok := prices[symbol]; ok {
			pos.unrealized_pnl = (price - pos.avg_price) * pos.quantity
		}
	}
	append(&p.equity_curve, portfolio_total_value(p, prices))
}

// ============================================================================
// Strategy Interface
// ============================================================================

StrategyContext :: struct {
	portfolio:   ^Portfolio,
	data:        ^w.DataFrame,
	current_bar: int,
	symbols:     []string,
	config:      BacktestConfig, // commission and slippage applied to fills (set by the runners)
}

context_get_price :: proc(ctx: ^StrategyContext, symbol: string) -> f64 {
	col := w.column(ctx.data, symbol)
	price, _ := w.column_at_float(col, ctx.current_bar)
	return price
}

context_get_prices :: proc(
	ctx: ^StrategyContext,
	symbols: []string,
	allocator: mem.Allocator,
) -> map[string]f64 {
	prices := make(map[string]f64, allocator)
	for symbol in symbols {
		prices[symbol] = context_get_price(ctx, symbol)
	}
	return prices
}

// Fills a market order at the current bar's price moved against the trader by
// config.slippage_rate, charging config.commission_rate on the fill notional.
// Buys reduce or cover a short before going long; sells reduce a long before
// going short. Orders with a non-positive quantity or price are ignored.
context_submit_order :: proc(ctx: ^StrategyContext, order: Order) {
	if order.order_type != .Market {return}
	price := context_get_price(ctx, order.symbol)
	if !(order.quantity > 0.0) ||
	   !(price > 0.0) ||
	   math.is_inf(price) ||
	   math.is_inf(order.quantity) {
		return
	}

	slip := ctx.config.slippage_rate
	price *= (1.0 + slip) if order.side == .Buy else (1.0 - slip)
	commission := order.quantity * price * ctx.config.commission_rate
	q := order.quantity if order.side == .Buy else -order.quantity

	pf := ctx.portfolio
	pf.cash -= q * price + commission

	pos, _ := pf.positions[order.symbol]
	pos.symbol = order.symbol
	Q := pos.quantity
	eps := 1e-9 * max(abs(Q), abs(q))

	if abs(Q) <= eps || (Q > 0.0) == (q > 0.0) {
		// Open or add: average the entry price
		if abs(Q) <= eps {
			Q = 0.0
			pos.trade_pnl = 0.0
		}
		pos.avg_price = (pos.avg_price * abs(Q) + price * abs(q)) / (abs(Q) + abs(q))
		pos.quantity = Q + q
		pos.trade_pnl -= commission
	} else {
		// Reduce, close, or flip
		close_qty := min(abs(q), abs(Q))
		sgn := 1.0 if Q > 0.0 else -1.0
		realized := (price - pos.avg_price) * close_qty * sgn
		close_comm := commission * close_qty / abs(q)
		pos.realized_pnl += realized
		pos.trade_pnl += realized - close_comm
		pos.quantity = Q - sgn * close_qty

		if abs(pos.quantity) <= eps {
			pos.quantity = 0.0
			pos.round_trips += 1
			if pos.trade_pnl > 0.0 {pos.winning_trades += 1}
			append(&pf.trade_pnls, pos.trade_pnl)
			pos.trade_pnl = 0.0
			if rest := abs(q) - close_qty; rest > eps {
				// Flip: the remainder opens a new trip in the order's direction
				pos.quantity = rest if q > 0.0 else -rest
				pos.avg_price = price
				pos.trade_pnl = -(commission - close_comm)
			}
		}
	}
	pos.is_short = pos.quantity < 0.0
	pf.positions[order.symbol] = pos

	append(
		&pf.trades,
		FilledOrder {
			order = order,
			fill_price = price,
			fill_time = ctx.current_bar,
			commission = commission,
		},
	)
}

StrategyFn :: proc(ctx: ^StrategyContext)

// ============================================================================
// Backtest Engine
// ============================================================================

// Returns are bar-to-bar returns of the equity curve, the first one measured
// from initial_capital; 252 bars per year. win_rate and profit_factor are over
// closed round trips, net of commissions (open positions are not counted);
// profit_factor is +Inf when there are winning trips but no losing ones.
BacktestResult :: struct {
	equity_curve:  []f64,
	trades:        []FilledOrder,
	total_return:  f64,
	annual_return: f64, // geometric, n_bars / 252 years
	sharpe_ratio:  f64,
	max_drawdown:  f64, // ≤ 0
	win_rate:      f64,
	profit_factor: f64,
	total_trades:  int, // fills
	round_trips:   int, // closed round trips (win_rate / profit_factor sample)
}


_calculate_backtest_metrics :: proc(
	portfolio: ^Portfolio,
	config: BacktestConfig,
	allocator: mem.Allocator,
) -> BacktestResult {
	result: BacktestResult

	result.equity_curve = make([]f64, len(portfolio.equity_curve), allocator)
	copy(result.equity_curve, portfolio.equity_curve[:])

	result.trades = make([]FilledOrder, len(portfolio.trades), allocator)
	copy(result.trades, portfolio.trades[:])
	result.total_trades = len(portfolio.trades)

	n := len(portfolio.equity_curve)
	if n == 0 || !(config.initial_capital > 0.0) {
		return result
	}

	returns := make([]f64, n, allocator)
	defer delete(returns, allocator)
	prev := config.initial_capital
	for i in 0 ..< n {
		e := portfolio.equity_curve[i]
		returns[i] = (e - prev) / prev
		prev = e
	}

	result.total_return = portfolio.equity_curve[n - 1] / config.initial_capital - 1.0
	n_years := f64(n) / 252.0
	result.annual_return =
		math.pow(1.0 + result.total_return, 1.0 / n_years) - 1.0 if result.total_return > -1.0 else -1.0

	result.sharpe_ratio = sharpe_ratio_from_returns(returns, config.risk_free_rate, 252.0)
	result.max_drawdown = max_drawdown(returns)

	wins := 0
	gross_profit, gross_loss := 0.0, 0.0
	for pnl in portfolio.trade_pnls {
		if pnl > 0.0 {
			wins += 1
			gross_profit += pnl
		} else {
			gross_loss -= pnl
		}
	}
	result.round_trips = len(portfolio.trade_pnls)
	if len(portfolio.trade_pnls) > 0 {
		result.win_rate = f64(wins) / f64(len(portfolio.trade_pnls))
	}
	if gross_loss > 0.0 {
		result.profit_factor = gross_profit / gross_loss
	} else if gross_profit > 0.0 {
		result.profit_factor = math.INF_F64
	}

	return result
}

// ============================================================================
// Visualization
// ============================================================================

plot_equity_curve :: proc(
	result: ^BacktestResult,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(result.equity_curve) == 0 {return false}

	xs := make([]f64, len(result.equity_curve), allocator)
	ys := make([]f64, len(result.equity_curve), allocator)
	defer {
		delete(xs, allocator)
		delete(ys, allocator)
	}

	for i in 0 ..< len(result.equity_curve) {
		xs[i] = f64(i)
		ys[i] = result.equity_curve[i]
	}

	lines := []p.LineData {
		p.LineData{xs = xs, ys = ys, color = p.BLUE, style = .Solid, label = "Equity"},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Backtest Equity Curve"
	config.y_label = "Portfolio Value ($)"
	config.x_label = "Trading Day"
	config.show_grid = true

	return p.multi_line_png(lines, path, config, allocator)
}

plot_backtest_drawdown :: proc(
	result: ^BacktestResult,
	path: string,
	allocator: mem.Allocator = context.allocator,
) -> bool {
	if len(result.equity_curve) < 2 {return false}

	n := len(result.equity_curve)
	xs := make([]f64, n, allocator)
	ys := make([]f64, n, allocator)
	defer {
		delete(xs, allocator)
		delete(ys, allocator)
	}

	peak := result.equity_curve[0]
	for i in 0 ..< n {
		xs[i] = f64(i)
		if result.equity_curve[i] > peak {
			peak = result.equity_curve[i]
		}
		ys[i] = (result.equity_curve[i] - peak) / peak * 100.0
	}

	lines := []p.LineData {
		p.LineData{xs = xs, ys = ys, color = p.RED, style = .Solid, label = "Drawdown %"},
	}

	config := p.DEFAULT_PLOT_CONFIG
	config.title = "Backtest Drawdown"
	config.y_label = "Drawdown (%)"
	config.x_label = "Trading Day"
	config.show_grid = true

	return p.multi_line_png(lines, path, config, allocator)
}

// ============================================================================
// Strategy 1: SMA Crossover
// ============================================================================

sma_crossover_strategy :: proc(
	ctx: ^StrategyContext,
	symbol: string,
	fast_period: int,
	slow_period: int,
	position_size: f64,
) {
	if ctx.current_bar < slow_period {return}

	col := w.column(ctx.data, symbol)

	// Calculate current SMAs
	fast_sum := 0.0
	for i in ctx.current_bar - fast_period + 1 ..= ctx.current_bar {
		price, _ := w.column_at_float(col, i)
		fast_sum += price
	}
	fast_sma := fast_sum / f64(fast_period)

	slow_sum := 0.0
	for i in ctx.current_bar - slow_period + 1 ..= ctx.current_bar {
		price, _ := w.column_at_float(col, i)
		slow_sum += price
	}
	slow_sma := slow_sum / f64(slow_period)

	// Check if we have a position WITH QUANTITY > 0
	has_position := false
	if pos, ok := ctx.portfolio^.positions[symbol]; ok {
		has_position = pos.quantity > 0.0001
	}

	// Get previous SMAs to detect crossover
	prev_fast_sum := 0.0
	for i in ctx.current_bar - fast_period ..= ctx.current_bar - 1 {
		price, _ := w.column_at_float(col, i)
		prev_fast_sum += price
	}
	prev_fast_sma := prev_fast_sum / f64(fast_period)

	prev_slow_sum := 0.0
	for i in ctx.current_bar - slow_period ..= ctx.current_bar - 1 {
		price, _ := w.column_at_float(col, i)
		prev_slow_sum += price
	}
	prev_slow_sma := prev_slow_sum / f64(slow_period)

	// Golden cross: fast crosses above slow
	golden_cross := (prev_fast_sma <= prev_slow_sma) && (fast_sma > slow_sma)

	// Death cross: fast crosses below slow
	death_cross := (prev_fast_sma >= prev_slow_sma) && (fast_sma < slow_sma)

	current_price := context_get_price(ctx, symbol)

	// Buy on golden cross if not in position
	if golden_cross && !has_position {
		quantity := (ctx.portfolio^.cash * position_size) / current_price
		order := Order {
			symbol     = symbol,
			side       = .Buy,
			quantity   = quantity,
			order_type = .Market,
		}
		context_submit_order(ctx, order)
	} else if death_cross && has_position {
		// Sell on death cross if in position
		if pos, ok := ctx.portfolio^.positions[symbol]; ok {
			order := Order {
				symbol     = symbol,
				side       = .Sell,
				quantity   = pos.quantity,
				order_type = .Market,
			}
			context_submit_order(ctx, order)
		}
	}
}

// ============================================================================
// Strategy 2: Mean Reversion (Bollinger Bands)
// ============================================================================

mean_reversion_strategy :: proc(
	ctx: ^StrategyContext,
	symbol: string,
	lookback: int,
	num_std_dev: f64,
	position_size: f64,
) {
	if ctx.current_bar < lookback {return}

	col := w.column(ctx.data, symbol)

	// Calculate mean and standard deviation
	sum := 0.0
	for i in ctx.current_bar - lookback + 1 ..= ctx.current_bar {
		price, _ := w.column_at_float(col, i)
		sum += price
	}
	mean := sum / f64(lookback)

	sum_sq := 0.0
	for i in ctx.current_bar - lookback + 1 ..= ctx.current_bar {
		price, _ := w.column_at_float(col, i)
		diff := price - mean
		sum_sq += diff * diff
	}
	std_dev := math.sqrt(sum_sq / f64(lookback))

	// Bollinger Bands
	upper_band := mean + num_std_dev * std_dev
	lower_band := mean - num_std_dev * std_dev

	current_price := context_get_price(ctx, symbol)

	// Check if we have a position
	has_position := false
	if pos, ok := ctx.portfolio^.positions[symbol]; ok {
		has_position = pos.quantity > 0.0001
	}

	// Buy when price touches lower band (oversold)
	if current_price <= lower_band && !has_position {
		quantity := (ctx.portfolio^.cash * position_size) / current_price
		order := Order {
			symbol     = symbol,
			side       = .Buy,
			quantity   = quantity,
			order_type = .Market,
		}
		context_submit_order(ctx, order)
	} else if current_price >= upper_band && has_position {
		// Sell when price touches upper band (overbought)
		if pos, ok := ctx.portfolio^.positions[symbol]; ok {
			order := Order {
				symbol     = symbol,
				side       = .Sell,
				quantity   = pos.quantity,
				order_type = .Market,
			}
			context_submit_order(ctx, order)
		}
	}
}

// ============================================================================
// Strategy 3: Momentum (Breakout)
// ============================================================================

momentum_strategy :: proc(
	ctx: ^StrategyContext,
	symbol: string,
	lookback: int,
	position_size: f64,
) {
	if ctx.current_bar < lookback {return}

	col := w.column(ctx.data, symbol)

	// Find highest high and lowest low over lookback period
	highest := -math.F64_MAX
	lowest := math.F64_MAX

	for i in ctx.current_bar - lookback ..= ctx.current_bar - 1 {
		price, _ := w.column_at_float(col, i)
		if price > highest {highest = price}
		if price < lowest {lowest = price}
	}

	current_price := context_get_price(ctx, symbol)

	// Check if we have a position
	has_position := false
	if pos, ok := ctx.portfolio^.positions[symbol]; ok {
		has_position = pos.quantity > 0.0001
	}

	// Buy on breakout above highest high
	if current_price > highest && !has_position {
		quantity := (ctx.portfolio^.cash * position_size) / current_price
		order := Order {
			symbol     = symbol,
			side       = .Buy,
			quantity   = quantity,
			order_type = .Market,
		}
		context_submit_order(ctx, order)
	} else if current_price < lowest && has_position {
		// Sell on breakdown below lowest low
		if pos, ok := ctx.portfolio^.positions[symbol]; ok {
			order := Order {
				symbol     = symbol,
				side       = .Sell,
				quantity   = pos.quantity,
				order_type = .Market,
			}
			context_submit_order(ctx, order)
		}
	}
}


// ============================================================================
// Strategy 4: Pairs Trading (Statistical Arbitrage)
// ============================================================================

pairs_trading_strategy :: proc(
	ctx: ^StrategyContext,
	symbol1: string,
	symbol2: string,
	lookback: int,
	entry_z_score: f64,
	exit_z_score: f64,
	position_size: f64,
) {
	if ctx.current_bar < lookback {return}

	col1 := w.column(ctx.data, symbol1)
	col2 := w.column(ctx.data, symbol2)

	// Calculate spread (price1 - price2)
	spreads := make([]f64, lookback)
	defer delete(spreads)
	for i in 0 ..< lookback {
		price1, _ := w.column_at_float(col1, ctx.current_bar - lookback + 1 + i)
		price2, _ := w.column_at_float(col2, ctx.current_bar - lookback + 1 + i)
		spreads[i] = price1 - price2
	}

	// Calculate mean and std of spread
	sum := 0.0
	for s in spreads {sum += s}
	spread_mean := sum / f64(lookback)

	sum_sq := 0.0
	for s in spreads {
		diff := s - spread_mean
		sum_sq += diff * diff
	}
	spread_std := math.sqrt(sum_sq / f64(lookback))

	if spread_std < 1e-10 {return}

	// Current spread
	price1 := context_get_price(ctx, symbol1)
	price2 := context_get_price(ctx, symbol2)
	current_spread := price1 - price2

	// Z-score
	z_score := (current_spread - spread_mean) / spread_std

	// Check positions
	has_pos1 := false
	has_pos2 := false
	if pos, ok := ctx.portfolio^.positions[symbol1]; ok {
		has_pos1 = pos.quantity > 0.0001
	}
	if pos, ok := ctx.portfolio^.positions[symbol2]; ok {
		has_pos2 = pos.quantity > 0.0001
	}

	// Entry: spread diverges (z_score > entry_z_score or < -entry_z_score)
	if z_score > entry_z_score && !has_pos1 && !has_pos2 {
		// Spread too high: short symbol1, long symbol2
		quantity1 := (ctx.portfolio^.cash * position_size * 0.5) / price1
		quantity2 := (ctx.portfolio^.cash * position_size * 0.5) / price2

		// Note: This simplified version only goes long symbol2
		// For true pairs trading, you'd need short selling support
		order := Order {
			symbol     = symbol2,
			side       = .Buy,
			quantity   = quantity2,
			order_type = .Market,
		}
		context_submit_order(ctx, order)
	} else if z_score < -entry_z_score && !has_pos1 && !has_pos2 {
		// Spread too low: long symbol1, short symbol2
		quantity1 := (ctx.portfolio^.cash * position_size * 0.5) / price1

		// Simplified: only long symbol1
		order := Order {
			symbol     = symbol1,
			side       = .Buy,
			quantity   = quantity1,
			order_type = .Market,
		}
		context_submit_order(ctx, order)
	} else if math.abs(z_score) < exit_z_score && (has_pos1 || has_pos2) {
		// Exit: spread reverts to mean
		if has_pos1 {
			if pos, ok := ctx.portfolio^.positions[symbol1]; ok {
				order := Order {
					symbol     = symbol1,
					side       = .Sell,
					quantity   = pos.quantity,
					order_type = .Market,
				}
				context_submit_order(ctx, order)
			}
		}
		if has_pos2 {
			if pos, ok := ctx.portfolio^.positions[symbol2]; ok {
				order := Order {
					symbol     = symbol2,
					side       = .Sell,
					quantity   = pos.quantity,
					order_type = .Market,
				}
				context_submit_order(ctx, order)
			}
		}
	}
}

pairs_trading_full :: proc(
	ctx: ^StrategyContext,
	symbol1: string,
	symbol2: string,
	lookback: int,
	entry_z_score: f64,
	exit_z_score: f64,
	position_size: f64,
) {
	if ctx.current_bar < lookback {return}

	col1 := w.column(ctx.data, symbol1)
	col2 := w.column(ctx.data, symbol2)

	// Calculate spread
	spreads := make([]f64, lookback, context.temp_allocator)
	defer delete(spreads, context.temp_allocator)
	for i in 0 ..< lookback {
		price1, _ := w.column_at_float(col1, ctx.current_bar - lookback + 1 + i)
		price2, _ := w.column_at_float(col2, ctx.current_bar - lookback + 1 + i)
		spreads[i] = price1 - price2
	}

	// Calculate mean and std
	sum := 0.0
	for s in spreads {sum += s}
	spread_mean := sum / f64(lookback)

	sum_sq := 0.0
	for s in spreads {
		diff := s - spread_mean
		sum_sq += diff * diff
	}
	spread_std := math.sqrt(sum_sq / f64(lookback))

	if spread_std < 1e-10 {return}

	// Current spread and z-score
	price1 := context_get_price(ctx, symbol1)
	price2 := context_get_price(ctx, symbol2)
	current_spread := price1 - price2
	z_score := (current_spread - spread_mean) / spread_std

	// Check positions
	has_pos1 := false
	has_pos2 := false
	if pos, ok := ctx.portfolio^.positions[symbol1]; ok {
		has_pos1 = math.abs(pos.quantity) > 0.0001
	}
	if pos, ok := ctx.portfolio^.positions[symbol2]; ok {
		has_pos2 = math.abs(pos.quantity) > 0.0001
	}

	// Entry signals
	if z_score > entry_z_score && !has_pos1 && !has_pos2 {
		// Spread too high: short symbol1, long symbol2
		quantity1 := (ctx.portfolio^.cash * position_size * 0.5) / price1
		quantity2 := (ctx.portfolio^.cash * position_size * 0.5) / price2

		// Short symbol1
		order1 := Order {
			symbol     = symbol1,
			side       = .Sell,
			quantity   = quantity1,
			order_type = .Market,
		}
		context_submit_order(ctx, order1)

		// Long symbol2
		order2 := Order {
			symbol     = symbol2,
			side       = .Buy,
			quantity   = quantity2,
			order_type = .Market,
		}
		context_submit_order(ctx, order2)
	} else if z_score < -entry_z_score && !has_pos1 && !has_pos2 {
		// Spread too low: long symbol1, short symbol2
		quantity1 := (ctx.portfolio^.cash * position_size * 0.5) / price1
		quantity2 := (ctx.portfolio^.cash * position_size * 0.5) / price2

		// Long symbol1
		order1 := Order {
			symbol     = symbol1,
			side       = .Buy,
			quantity   = quantity1,
			order_type = .Market,
		}
		context_submit_order(ctx, order1)

		// Short symbol2
		order2 := Order {
			symbol     = symbol2,
			side       = .Sell,
			quantity   = quantity2,
			order_type = .Market,
		}
		context_submit_order(ctx, order2)
	} else if math.abs(z_score) < exit_z_score && (has_pos1 || has_pos2) {
		// Exit: spread reverts to mean - close all positions
		if has_pos1 {
			if pos, ok := ctx.portfolio^.positions[symbol1]; ok {
				side := OrderSide.Sell
				if pos.is_short {
					side = .Buy
				}
				order := Order {
					symbol     = symbol1,
					side       = side,
					quantity   = math.abs(pos.quantity),
					order_type = .Market,
				}
				context_submit_order(ctx, order)
			}
		}
		if has_pos2 {
			if pos, ok := ctx.portfolio^.positions[symbol2]; ok {
				side := OrderSide.Sell
				if pos.is_short {
					side = .Buy
				}
				order := Order {
					symbol     = symbol2,
					side       = side,
					quantity   = math.abs(pos.quantity),
					order_type = .Market,
				}
				context_submit_order(ctx, order)
			}
		}
	}
}

PortfolioStrategyFn :: proc(ctx: ^StrategyContext, allocator: mem.Allocator) -> map[string]f64
backtest_run :: proc(
	data: ^w.DataFrame,
	symbols: []string,
	strategy: StrategyFn,
	config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> BacktestResult {
	portfolio := portfolio_init(config.initial_capital, allocator)

	ctx := StrategyContext {
		portfolio   = &portfolio,
		data        = data,
		current_bar = 0,
		symbols     = symbols,
		config      = config,
	}

	n_bars := data.rows
	for bar in 0 ..< n_bars {
		ctx.current_bar = bar

		prices := context_get_prices(&ctx, symbols, allocator)

		strategy(&ctx)

		portfolio_update(&portfolio, prices)

		delete(prices)
	}

	result := _calculate_backtest_metrics(&portfolio, config, allocator)
	_portfolio_destroy(&portfolio)
	return result
}

backtest_portfolio_run :: proc(
	data: ^w.DataFrame,
	symbols: []string,
	strategy: PortfolioStrategyFn,
	rebalance_frequency: int = 21,
	config: BacktestConfig = DEFAULT_BACKTEST_CONFIG,
	allocator: mem.Allocator = context.allocator,
) -> BacktestResult {
	portfolio := portfolio_init(config.initial_capital, allocator)

	ctx := StrategyContext {
		portfolio   = &portfolio,
		data        = data,
		current_bar = 0,
		symbols     = symbols,
		config      = config,
	}

	n_bars := data.rows
	for bar in 0 ..< n_bars {
		ctx.current_bar = bar

		prices := context_get_prices(&ctx, symbols, allocator)

		if bar % rebalance_frequency == 0 {
			target_weights := strategy(&ctx, allocator)

			total_value := portfolio_total_value(&portfolio, prices)

			// Sells first so the buys are funded by their proceeds; drifts below
			// 1% of the portfolio value are not traded. Weights < 0 only sell
			// down to flat (no shorting).
			for pass in 0 ..< 2 {
				for sym in symbols {
					target_w := target_weights[sym] or_else 0.0
					current_price := prices[sym]
					if !(current_price > 0.0) {continue}
					current_qty := 0.0
					if pos, ok := portfolio.positions[sym]; ok {current_qty = pos.quantity}

					delta_value := total_value * max(target_w, 0.0) - current_qty * current_price
					if abs(delta_value) <= total_value * 0.01 {continue}
					is_sell := delta_value < 0.0
					if is_sell != (pass == 0) {continue}

					qty := abs(delta_value) / current_price
					if is_sell {qty = min(qty, max(current_qty, 0.0))}
					if qty > 1e-9 {
						context_submit_order(
							&ctx,
							Order {
								symbol = sym,
								side = .Sell if is_sell else .Buy,
								quantity = qty,
								order_type = .Market,
							},
						)
					}
				}
			}

			delete(target_weights)
		}

		portfolio_update(&portfolio, prices)

		delete(prices)
	}

	result := _calculate_backtest_metrics(&portfolio, config, allocator)
	_portfolio_destroy(&portfolio)
	return result
}
// ============================================================================
// Portfolio Strategy 1: Equal Weight (Periodic Rebalance)
// ============================================================================

portfolio_equal_weight :: proc(ctx: ^StrategyContext, allocator: mem.Allocator) -> map[string]f64 {
	weights := make(map[string]f64, allocator)
	n := f64(len(ctx.symbols))

	// Allocate 95% to assets, keep 5% cash buffer for slippage/commissions
	allocation := 0.95 / n
	for sym in ctx.symbols {
		weights[sym] = allocation
	}
	return weights
}

// ============================================================================
// Portfolio Strategy 2: Inverse Volatility (Risk Parity Lite)
// ============================================================================
// Allocates more capital to lower-volatility assets to equalize risk contribution.

portfolio_inverse_volatility :: proc(
	ctx: ^StrategyContext,
	allocator: mem.Allocator,
	lookback: int = 60,
) -> map[string]f64 {
	weights := make(map[string]f64, allocator)

	if ctx.current_bar < lookback {
		n := f64(len(ctx.symbols))
		for sym in ctx.symbols {weights[sym] = 0.95 / n}
		return weights
	}

	inv_vols := make([]f64, len(ctx.symbols), context.temp_allocator)
	defer delete(inv_vols, context.temp_allocator) // ← ADD THIS

	sum_inv_vol := 0.0

	for sym, i in ctx.symbols {
		col := w.column(ctx.data, sym)

		sum_ret := 0.0
		sum_sq := 0.0
		count := 0

		for j in ctx.current_bar - lookback + 1 ..= ctx.current_bar {
			p1, _ := w.column_at_float(col, j - 1)
			p2, _ := w.column_at_float(col, j)
			if p1 > 0 {
				ret := (p2 - p1) / p1
				sum_ret += ret
				sum_sq += ret * ret
				count += 1
			}
		}

		if count > 1 {
			mean_ret := sum_ret / f64(count)
			variance := (sum_sq / f64(count)) - (mean_ret * mean_ret)
			vol := math.sqrt(math.max(0.0, variance))

			if vol > 1e-8 {
				inv_vols[i] = 1.0 / vol
				sum_inv_vol += inv_vols[i]
			}
		}
	}

	if sum_inv_vol > 0 {
		for sym, i in ctx.symbols {
			weights[sym] = 0.95 * (inv_vols[i] / sum_inv_vol)
		}
	} else {
		n := f64(len(ctx.symbols))
		for sym in ctx.symbols {weights[sym] = 0.95 / n}
	}

	return weights
}

// ============================================================================
// Portfolio Strategy 3: Rolling Minimum Variance (Markowitz)
// ============================================================================
portfolio_min_variance :: proc(
	ctx: ^StrategyContext,
	allocator: mem.Allocator,
	lookback: int = 120,
) -> map[string]f64 {
	weights := make(map[string]f64, allocator)
	n_assets := len(ctx.symbols)

	if ctx.current_bar < lookback {
		for sym in ctx.symbols {weights[sym] = 0.95 / f64(n_assets)}
		return weights
	}

	// 1. Returns over the last `lookback` bars, ending at the current bar
	//    (observations × assets)
	R := make([]f64, lookback * n_assets, context.temp_allocator)
	defer delete(R, context.temp_allocator)
	for i in 0 ..< n_assets {
		col := w.column(ctx.data, ctx.symbols[i])
		for j in 0 ..< lookback {
			bar := ctx.current_bar - lookback + 1 + j
			p1, _ := w.column_at_float(col, bar - 1)
			p2, _ := w.column_at_float(col, bar)
			R[j * n_assets + i] = (p2 - p1) / p1 if p1 > 0 else 0.0
		}
	}

	// 2. Sample covariance between assets
	cov_linalg := l.matrix_new(f64, n_assets, n_assets, context.temp_allocator)
	defer l.matrix_free(&cov_linalg)
	means := make([]f64, n_assets, context.temp_allocator)
	defer delete(means, context.temp_allocator)
	for j in 0 ..< lookback {
		for i in 0 ..< n_assets {means[i] += R[j * n_assets + i] / f64(lookback)}
	}
	for i in 0 ..< n_assets {
		for k in i ..< n_assets {
			c := 0.0
			for j in 0 ..< lookback {
				c += (R[j * n_assets + i] - means[i]) * (R[j * n_assets + k] - means[k])
			}
			c /= f64(max(lookback - 1, 1))
			cov_linalg.data[i * n_assets + k] = c
			cov_linalg.data[k * n_assets + i] = c
		}
	}

	// 3. Optimize
	constraints := PortfolioConstraints {
		no_short_selling = true,
		max_weight       = 0.0,
	}

	opt_weights := constrained_min_variance_portfolio(
		&cov_linalg,
		constraints,
		context.temp_allocator,
	)
	defer delete(opt_weights, context.temp_allocator)

	// 4. Map back to symbols
	for i in 0 ..< n_assets {
		weights[ctx.symbols[i]] = opt_weights[i] * 0.95
	}

	return weights
}
