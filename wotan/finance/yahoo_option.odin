package finance

import w "../core"
import net "../net"

import "core:fmt"
import "core:math"
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

// Main function to fetch and parse Yahoo Options
fetch_yahoo_options :: proc(
	symbol: string,
	allocator: mem.Allocator = context.allocator,
) -> OptionChain {
	// 1. Get crumb and cookie first
	crumb, cookie, ok := net.yahoo_get_crumb_and_cookie(allocator)
	if !ok {
		fmt.println("Warning: Failed to get Yahoo crumb/cookie")
		return OptionChain{}
	}

	// 2. Build URL with crumb
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/options/%s?crumb=%s",
		symbol,
		crumb,
		allocator = allocator,
	)
	defer delete(url, allocator)

	// 3. Fetch using your EXISTING net.http_get_with_cookie
	text, ok2 := net.http_get_with_cookie(url, cookie, allocator)
	if !ok2 {
		panic(fmt.tprintf("Failed to fetch options data for %s", symbol))
	}
	defer delete(text, allocator)

	// ========================================================================
	// KEEP YOUR EXISTING PARSING LOGIC EXACTLY AS IT IS BELOW THIS LINE
	// ========================================================================

	// Dynamic arrays for building the chain
	strikes := make([dynamic]f64, allocator)
	expiries := make([dynamic]f64, allocator)
	implied_vols := make([dynamic]f64, allocator)
	market_prices := make([dynamic]f64, allocator)
	option_types := make([dynamic]string, allocator)

	defer {
		delete(strikes)
		delete(expiries)
		delete(implied_vols)
		delete(market_prices)
		delete(option_types)
	}

	// Find the options block
	opt_pos := strings.index(text, "\"options\":")
	if opt_pos < 0 {
		fmt.println("Warning: No options data found in Yahoo response")
		return OptionChain{}
	}


	// Extract expiration date
	exp_timestamp, _ := _parse_json_f64(text, "\"expirationDate\":", opt_pos)

	// Extract current market time for time-to-expiry calculation
	// Extract current market time for time-to-expiry calculation
	current_timestamp, _ := _parse_json_f64(text, "\"regularMarketTime\":", opt_pos)
	if current_timestamp == 0.0 {
		// Fallback to system time - use now_unix() which returns i64 directly
		current_timestamp = f64(w.now_unix())
	}

	// Parse Calls and Puts
	_extract_options_array(
		text,
		"\"calls\":",
		"call",
		exp_timestamp,
		current_timestamp,
		&strikes,
		&expiries,
		&implied_vols,
		&market_prices,
		&option_types,
	)
	_extract_options_array(
		text,
		"\"puts\":",
		"put",
		exp_timestamp,
		current_timestamp,
		&strikes,
		&expiries,
		&implied_vols,
		&market_prices,
		&option_types,
	)

	// Convert dynamic arrays to fixed slices for OptionChain
	n := len(strikes)
	if n == 0 {
		return OptionChain{}
	}

	final_strikes := make([]f64, n, allocator)
	final_expiries := make([]f64, n, allocator)
	final_ivs := make([]f64, n, allocator)
	final_prices := make([]f64, n, allocator)
	final_types := make([]string, n, allocator)

	for i in 0 ..< n {
		final_strikes[i] = strikes[i]
		final_expiries[i] = expiries[i]
		final_ivs[i] = implied_vols[i]
		final_prices[i] = market_prices[i]
		final_types[i] = option_types[i]
	}

	return OptionChain {
		strikes = final_strikes,
		expiries = final_expiries,
		implied_vols = final_ivs,
		market_prices = final_prices,
		option_types = final_types,
		n_options = n,
	}
}
