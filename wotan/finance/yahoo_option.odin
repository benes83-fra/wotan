package finance

import net "../net"

import "core:fmt"
import "core:mem"
import "core:strconv"
import "core:strings"
import "core:time"

// ============================================================================
// Yahoo Finance Options Data Fetcher
// ============================================================================

// Helper to parse a float64 from JSON after a specific key
_parse_json_f64 :: proc(json: string, key: string, start_pos: int) -> (f64, int) {
	pos := strings.index(json[start_pos:], key)
	if pos < 0 {return 0.0, start_pos}
	pos += start_pos + len(key)

	// Skip whitespace and colon
	for pos < len(json) &&
	    (json[pos] == ' ' || json[pos] == ':' || json[pos] == '\t' || json[pos] == '\n') {
		pos += 1
	}

	// Handle null
	if pos + 4 <= len(json) && json[pos:pos + 4] == "null" {
		return 0.0, pos + 4
	}

	// Parse number
	end_pos := pos
	for end_pos < len(json) &&
	    (json[end_pos] >= '0' && json[end_pos] <= '9' ||
			    json[end_pos] == '.' ||
			    json[end_pos] == '-' ||
			    json[end_pos] == 'e' ||
			    json[end_pos] == 'E' ||
			    json[end_pos] == '+') {
		end_pos += 1
	}

	val, ok := strconv.parse_f64(json[pos:end_pos])
	if !ok {return 0.0, end_pos}
	return val, end_pos
}

// Extract options from a specific array (calls or puts)
_extract_options_array :: proc(
	json: string,
	array_name: string,
	opt_type: string,
	exp_timestamp: f64,
	current_timestamp: f64,
	strikes: ^[dynamic]f64,
	expiries: ^[dynamic]f64,
	implied_vols: ^[dynamic]f64,
	market_prices: ^[dynamic]f64,
	option_types: ^[dynamic]string,
) {
	pos := strings.index(json, array_name)
	if pos < 0 {return}

	bracket_pos := strings.index(json[pos:], "[")
	if bracket_pos < 0 {return}
	bracket_pos += pos

	end_bracket := strings.index(json[bracket_pos:], "]")
	if end_bracket < 0 {return}
	end_bracket += bracket_pos

	block := json[bracket_pos + 1:end_bracket]

	search_start := 0
	for {
		obj_start := strings.index(block[search_start:], "{")
		if obj_start < 0 {break}
		obj_start += search_start

		obj_end := strings.index(block[obj_start:], "}")
		if obj_end < 0 {break}
		obj_end += obj_start

		obj_str := block[obj_start:obj_end + 1]

		strike, _ := _parse_json_f64(obj_str, "\"strike\":", 0)
		last_price, _ := _parse_json_f64(obj_str, "\"lastPrice\":", 0)

		// Fallback to bid if lastPrice is 0 (illiquid)
		if last_price == 0.0 {
			bid, _ := _parse_json_f64(obj_str, "\"bid\":", 0)
			ask, _ := _parse_json_f64(obj_str, "\"ask\":", 0)
			if bid > 0.0 && ask > 0.0 {
				last_price = (bid + ask) / 2.0
			} else if bid > 0.0 {
				last_price = bid
			}
		}

		iv, _ := _parse_json_f64(obj_str, "\"impliedVolatility\":", 0)

		// Filter out invalid or deeply OTM options with 0 price
		if strike > 0.0 && last_price > 0.0 {
			append(strikes, strike)

			// Calculate time to expiry in years
			time_to_exp := (exp_timestamp - current_timestamp) / (365.25 * 24.0 * 3600.0)
			if time_to_exp < 0.0 {time_to_exp = 0.0}

			append(expiries, time_to_exp)

			append(implied_vols, iv)
			append(market_prices, last_price)
			append(option_types, opt_type)
		}

		search_start = obj_end + 1
	}
}


// ============================================================================
// Expiration dates
// ============================================================================
// The "expirationDates" of an options response (unix seconds, ascending)
_yahoo_expiration_dates :: proc(json: string, allocator: mem.Allocator) -> []f64 {
	out := make([dynamic]f64, allocator)
	pos := strings.index(json, "\"expirationDates\":")
	if pos < 0 {return out[:]}
	start := strings.index(json[pos:], "[")
	if start < 0 {return out[:]}
	start += pos
	end := strings.index(json[start:], "]")
	if end < 0 {return out[:]}
	block := json[start + 1:start + end]
	for e in strings.split_iterator(&block, ",") {
		if ts, ok := strconv.parse_f64(strings.trim_space(e)); ok {append(&out, ts)}
	}
	return out[:]
}

// The first expiration between min_days and max_days ahead, else the first future one
// (0 when there is none)
_find_target_expiration :: proc(
	json: string,
	current_timestamp: f64,
	min_days: f64,
	max_days: f64,
	allocator: mem.Allocator,
) -> f64 {
	dates := _yahoo_expiration_dates(json, allocator)
	defer delete(dates, allocator)
	min_ts := current_timestamp + min_days * 86400.0
	max_ts := current_timestamp + max_days * 86400.0
	for ts in dates {if ts >= min_ts && ts <= max_ts {return ts}}
	for ts in dates {if ts > current_timestamp {return ts}}
	return 0.0
}

// ============================================================================
// Fetching
// ============================================================================
@(private = "file")
_Chain_Builder :: struct {
	strikes, expiries, implied_vols, market_prices: [dynamic]f64,
	option_types:                                   [dynamic]string,
}

// Appends the calls and puts of one expiration date; false when the request fails
@(private = "file")
_fetch_yahoo_expiry :: proc(
	symbol, crumb, cookie: string,
	exp_ts, now_ts: f64,
	b: ^_Chain_Builder,
	allocator: mem.Allocator,
) -> bool {
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/options/%s?date=%d&crumb=%s",
		symbol,
		i64(exp_ts),
		crumb,
		allocator = allocator,
	)
	defer delete(url, allocator)
	text, ok := net.http_get_with_cookie(url, cookie, allocator)
	if !ok {return false}
	defer delete(text, allocator)
	opt_pos := strings.index(text, "\"options\":")
	if opt_pos < 0 {return false}
	exp_timestamp, _ := _parse_json_f64(text, "\"expirationDate\"", opt_pos)
	if exp_timestamp <= 0 {exp_timestamp = exp_ts}
	for side in ([2][2]string{{"\"calls\":", "call"}, {"\"puts\":", "put"}}) {
		_extract_options_array(
			text,
			side[0],
			side[1],
			exp_timestamp,
			now_ts,
			&b.strikes,
			&b.expiries,
			&b.implied_vols,
			&b.market_prices,
			&b.option_types,
		)
	}
	return true
}

// Crumb, cookie, the initial response (expiration dates) and the market time
@(private = "file")
_yahoo_options_session :: proc(
	symbol: string,
	allocator: mem.Allocator,
) -> (
	crumb, cookie, text: string,
	now_ts: f64,
	ok: bool,
) {
	crumb, cookie, ok = net.yahoo_get_crumb_and_cookie(allocator)
	if !ok {return}
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/options/%s?crumb=%s",
		symbol,
		crumb,
		allocator = allocator,
	)
	defer delete(url, allocator)
	text, ok = net.http_get_with_cookie(url, cookie, allocator)
	if !ok {return}
	now_ts, _ = _parse_json_f64(text, "\"regularMarketTime\"", 0)
	if now_ts <= 0 {now_ts = f64(time.now()._nsec) / 1e9}
	return
}

@(private = "file")
_chain_from_builder :: proc(b: ^_Chain_Builder, allocator: mem.Allocator) -> OptionChain {
	defer {
		delete(b.strikes)
		delete(b.expiries)
		delete(b.implied_vols)
		delete(b.market_prices)
		delete(b.option_types)
	}
	n := len(b.strikes)
	if n == 0 {return {}}
	clone :: proc(v: [dynamic]$T, allocator: mem.Allocator) -> []T {
		out := make([]T, len(v), allocator)
		copy(out, v[:])
		return out
	}
	return OptionChain {
		strikes = clone(b.strikes, allocator),
		expiries = clone(b.expiries, allocator),
		implied_vols = clone(b.implied_vols, allocator),
		market_prices = clone(b.market_prices, allocator),
		option_types = clone(b.option_types, allocator),
		n_options = n,
	}
}

// The chain of one expiration ~30-90 days out (else the first future one); expiries in
// years from the market time. ok is false when Yahoo cannot be reached or returns
// no options (it rate-limits heavy use); the chain is then empty
fetch_yahoo_options :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	chain: OptionChain,
	ok: bool,
) {
	crumb, cookie, text, now_ts, got := _yahoo_options_session(symbol, allocator)
	if !got {return {}, false}
	defer delete(text, allocator)
	target := _find_target_expiration(text, now_ts, 30.0, 90.0, context.temp_allocator)
	if target == 0.0 {return {}, false}
	b := _Chain_Builder {
		strikes       = make([dynamic]f64, allocator),
		expiries      = make([dynamic]f64, allocator),
		implied_vols  = make([dynamic]f64, allocator),
		market_prices = make([dynamic]f64, allocator),
		option_types  = make([dynamic]string, allocator),
	}
	_fetch_yahoo_expiry(symbol, crumb, cookie, target, now_ts, &b, allocator)
	chain = _chain_from_builder(&b, allocator)
	return chain, chain.n_options > 0
}

// The chains of every expiration between min_days and max_days ahead (at most
// max_expiries of them, spread evenly over the range): a volatility surface. ok is
// false when Yahoo cannot be reached or returns no options; the chain is then empty
fetch_yahoo_option_surface :: proc(
	symbol: string,
	min_days: f64 = 7,
	max_days: f64 = 800,
	max_expiries: int = 12,
	allocator: mem.Allocator = context.allocator,
) -> (
	chain: OptionChain,
	ok: bool,
) {
	crumb, cookie, text, now_ts, got := _yahoo_options_session(symbol, allocator)
	if !got {return {}, false}
	defer delete(text, allocator)
	dates := _yahoo_expiration_dates(text, allocator)
	defer delete(dates, allocator)
	in_range := make([dynamic]f64, allocator)
	defer delete(in_range)
	for ts in dates {
		days := (ts - now_ts) / 86400.0
		if days >= min_days && days <= max_days {append(&in_range, ts)}
	}
	b := _Chain_Builder {
		strikes       = make([dynamic]f64, allocator),
		expiries      = make([dynamic]f64, allocator),
		implied_vols  = make([dynamic]f64, allocator),
		market_prices = make([dynamic]f64, allocator),
		option_types  = make([dynamic]string, allocator),
	}
	n := len(in_range)
	picks := min(n, max_expiries)
	for k in 0 ..< picks {
		idx := k if picks == n else int(f64(k) * f64(n - 1) / f64(max(picks - 1, 1)) + 0.5)
		if k > 0 &&
		   picks < n &&
		   idx == int(f64(k - 1) * f64(n - 1) / f64(max(picks - 1, 1)) + 0.5) {continue}
		_fetch_yahoo_expiry(symbol, crumb, cookie, in_range[idx], now_ts, &b, allocator)
	}
	chain = _chain_from_builder(&b, allocator)
	return chain, chain.n_options > 0
}
