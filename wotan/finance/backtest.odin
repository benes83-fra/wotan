// wotan/finance/backtest.odin
package finance

import w "../core"
import l "../linalg"
import p "../plot"
import "core:fmt"
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

Position :: struct {
	symbol:         string,
	quantity:       f64,
	avg_price:      f64,
	unrealized_pnl: f64,
	realized_pnl:   f64,
}

// ============================================================================
// Portfolio State
// ============================================================================

Portfolio :: struct {
	cash:         f64,
	positions:    map[string]Position,
	equity_curve: [dynamic]f64, // Changed to [dynamic]
	trades:       [dynamic]FilledOrder, // Changed to [dynamic]
	current_bar:  int,
}

portfolio_init :: proc(capital: f64, allocator: mem.Allocator) -> Portfolio {
	return Portfolio {
		cash = capital,
		positions = make(map[string]Position, allocator),
		equity_curve = make([dynamic]f64, 0, allocator),
		trades = make([dynamic]FilledOrder, 0, allocator),
		current_bar = 0,
	}
}

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
			p.positions[symbol] = pos
		}
	}

	equity := portfolio_total_value(p, prices)
	append(&p.equity_curve, equity)
}

// ============================================================================
// Strategy Interface
// ============================================================================

StrategyContext :: struct {
	portfolio:   ^Portfolio,
	data:        ^w.DataFrame,
	current_bar: int,
	symbols:     []string,
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

context_submit_order :: proc(ctx: ^StrategyContext, order: Order) {
	if order.order_type == .Market {
		price := context_get_price(ctx, order.symbol)

		// Apply slippage
		if order.side == .Buy {
			price *= 1.0005
		} else {
			price *= 0.9995
		}

		// Calculate commission
		notional := order.quantity * price
		commission := notional * 0.001

		// Update portfolio
		if order.side == .Buy {
			ctx.portfolio^.cash -= (notional + commission)

			if pos, ok := ctx.portfolio^.positions[order.symbol]; ok {
				total_qty := pos.quantity + order.quantity
				avg_price := (pos.avg_price * pos.quantity + price * order.quantity) / total_qty
				pos.quantity = total_qty
				pos.avg_price = avg_price
				ctx.portfolio^.positions[order.symbol] = pos
			} else {
				ctx.portfolio^.positions[order.symbol] = Position {
					symbol    = order.symbol,
					quantity  = order.quantity,
					avg_price = price,
				}
			}
		} else {
			ctx.portfolio^.cash += (notional - commission)

			if pos, ok := ctx.portfolio^.positions[order.symbol]; ok {
				realized := (price - pos.avg_price) * order.quantity
				pos.realized_pnl += realized
				pos.quantity -= order.quantity
				if pos.quantity <= 0.0001 {
					// Mark position as closed (don't delete from map)
					pos.quantity = 0.0
					ctx.portfolio^.positions[order.symbol] = pos
				} else {
					ctx.portfolio^.positions[order.symbol] = pos
				}
			}
		}

		// Record trade
		append(
			&ctx.portfolio^.trades,
			FilledOrder {
				order = order,
				fill_price = price,
				fill_time = ctx.current_bar,
				commission = commission,
			},
		)
	}
}

StrategyFn :: proc(ctx: ^StrategyContext)

// ============================================================================
// Backtest Engine
// ============================================================================

BacktestResult :: struct {
	equity_curve:  []f64,
	trades:        []FilledOrder,
	total_return:  f64,
	annual_return: f64,
	sharpe_ratio:  f64,
	max_drawdown:  f64,
	win_rate:      f64,
	profit_factor: f64,
	total_trades:  int,
}

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
	}

	n_bars := data.rows
	for bar in 0 ..< n_bars {
		ctx.current_bar = bar

		prices := context_get_prices(&ctx, symbols, allocator)
		defer delete(prices)

		strategy(&ctx)

		portfolio_update(&portfolio, prices)
	}

	result := _calculate_backtest_metrics(&portfolio, config, allocator)

	// Cleanup
	delete(portfolio.positions)
	delete(portfolio.equity_curve)
	delete(portfolio.trades)

	return result
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

	if len(portfolio.equity_curve) < 2 {
		return result
	}

	n := len(portfolio.equity_curve)
	returns := make([]f64, n - 1, allocator)
	defer delete(returns, allocator)

	for i in 0 ..< n - 1 {
		returns[i] =
			(portfolio.equity_curve[i + 1] - portfolio.equity_curve[i]) / portfolio.equity_curve[i]
	}

	result.total_return =
		(portfolio.equity_curve[n - 1] - config.initial_capital) / config.initial_capital

	n_years := f64(n) / 252.0
	result.annual_return = math.pow(1.0 + result.total_return, 1.0 / n_years) - 1.0

	result.sharpe_ratio = sharpe_ratio_from_returns(returns, config.risk_free_rate, 252.0)
	result.max_drawdown = max_drawdown(returns)

	wins := 0
	gross_profit := 0.0
	gross_loss := 0.0

	for trade in portfolio.trades {
		pnl := 0.0
		if trade.order.side == .Buy {
			for other in portfolio.trades {
				if other.order.symbol == trade.order.symbol &&
				   other.order.side == .Sell &&
				   other.fill_time > trade.fill_time {
					pnl = (other.fill_price - trade.fill_price) * trade.order.quantity
					break
				}
			}
		}

		if pnl > 0 {
			wins += 1
			gross_profit += pnl
		} else if pnl < 0 {
			gross_loss += math.abs(pnl)
		}
	}

	if result.total_trades > 0 {
		result.win_rate = f64(wins) / f64(result.total_trades / 2)
	}

	if gross_loss > 0 {
		result.profit_factor = gross_profit / gross_loss
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
	result: ^BacktestResult, // Renamed to avoid conflict
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
// Example Strategies
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

	current_price := context_get_price(ctx, symbol)
	quantity := (ctx.portfolio^.cash * position_size) / current_price

	if _, has_position := ctx.portfolio^.positions[symbol]; !has_position {
		if fast_sma > slow_sma {
			order := Order {
				symbol     = symbol,
				side       = .Buy,
				quantity   = quantity,
				order_type = .Market,
			}
			context_submit_order(ctx, order)
		}
	} else {
		if fast_sma < slow_sma {
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
}
