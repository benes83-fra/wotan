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

			// ✅ DEBUG: Print exactly what this function sees for the first option
			if strike == 110.0 {
				fmt.printf(
					"   DEBUG _extract: exp=%.0f, curr=%.0f, time_to_exp=%.4f\n",
					exp_timestamp,
					current_timestamp,
					time_to_exp,
				)
			}

			append(expiries, time_to_exp)

			append(implied_vols, iv)
			append(market_prices, last_price)
			append(option_types, opt_type)
		}

		search_start = obj_end + 1
	}
}


// ============================================================================
// Helper to find a target expiration timestamp within a day range
// ============================================================================
_find_target_expiration :: proc(
	json: string,
	current_timestamp: f64,
	min_days: f64,
	max_days: f64,
	allocator: mem.Allocator,
) -> f64 {
	// Find the key
	pos := strings.index(json, "\"expirationDates\":")
	if pos < 0 {return 0.0}

	// Find the opening bracket
	start := strings.index(json[pos:], "[")
	if start < 0 {return 0.0}
	start += pos

	// Find the closing bracket
	end := strings.index(json[start:], "]")
	if end < 0 {return 0.0}
	end += start

	// Extract just the numbers, excluding the brackets
	block := json[start + 1:end]

	entries := strings.split(block, ",", allocator)
	defer {
		for e in entries {delete(e, allocator)}
		delete(entries, allocator)
	}

	min_ts := current_timestamp + (min_days * 24.0 * 3600.0)
	max_ts := current_timestamp + (max_days * 24.0 * 3600.0)

	// First pass: look for ideal range (e.g., 30-90 days)
	for e in entries {
		clean_e := strings.trim_space(e)
		ts, ok := strconv.parse_f64(clean_e)
		if ok && ts >= min_ts && ts <= max_ts {
			return ts
		}
	}

	// Second pass: fallback to any future date if none in ideal range
	for e in entries {
		clean_e := strings.trim_space(e)
		ts, ok := strconv.parse_f64(clean_e)
		if ok && ts > current_timestamp {
			return ts
		}
	}

	return 0.0
}

// ============================================================================
// Main function to fetch and parse Yahoo Options
// ============================================================================
// ============================================================================
// Main function to fetch and parse Yahoo Options
// ============================================================================
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

	// 2. Initial fetch to get available expiration dates
	url_initial := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/options/%s?crumb=%s",
		symbol,
		crumb,
		allocator = allocator,
	)

	text_initial, ok2 := net.http_get_with_cookie(url_initial, cookie, allocator)
	if !ok2 {
		panic(fmt.tprintf("Failed to fetch initial options data for %s", symbol))
	}
	defer delete(text_initial, allocator)

	// Extract current market time
	current_timestamp, _ := _parse_json_f64(text_initial, "\"regularMarketTime\"", 0)

	// ✅ FIXED: If parsing failed or returned 0, use the standard library system time
	// This bypasses any bugs in w.now_unix()
	if current_timestamp <= 0.0 {
		current_timestamp = f64(w.now_unix())
		fmt.printf(
			"   DEBUG: Parsed regularMarketTime was 0. Fallback to system time: %.0f\n",
			current_timestamp,
		)
	} else {
		fmt.printf("   DEBUG: Parsed current_timestamp = %.0f\n", current_timestamp)
	}

	// Find a target expiration date ~30-90 days out
	target_exp := _find_target_expiration(
		text_initial,
		current_timestamp,
		30.0,
		90.0,
		context.temp_allocator,
	)

	dte_days := (target_exp - current_timestamp) / (24.0 * 3600.0)
	fmt.printf("   DEBUG: Found target_exp = %.0f (DTE: %.1f days)\n", target_exp, dte_days)

	if target_exp == 0.0 {
		fmt.println("   Warning: Could not find a valid future expiration date.")
		return OptionChain{}
	}

	// 3. Fetch the specific expiration chain using the ?date= parameter
	url := fmt.aprintf(
		"https://query1.finance.yahoo.com/v7/finance/options/%s?date=%d&crumb=%s",
		symbol,
		i64(target_exp),
		crumb,
		allocator = allocator,
	)

	fmt.printf("   DEBUG: Requesting specific date URL: %s\n", url)

	text, ok3 := net.http_get_with_cookie(url, cookie, allocator)
	if !ok3 {
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

	// ✅ FIXED: Removed the colon from the key. _parse_json_f64 handles skipping it.
	exp_timestamp, _ := _parse_json_f64(text, "\"expirationDate\"", opt_pos)
	fmt.printf("   DEBUG: Parsed exp_timestamp = %.0f\n", exp_timestamp)

	// ✅ FIXED: Removed the colon from the key.
	current_timestamp, _ = _parse_json_f64(text, "\"regularMarketTime\"", opt_pos)
	fmt.printf("   DEBUG: Parsed current_timestamp (2nd) = %.0f\n", current_timestamp)

	// ✅ CRITICAL FIX: Replace w.now_unix() with this exact line
	if current_timestamp == 0.0 {
		current_timestamp = f64(time.now()._nsec) / 1_000_000_000.0
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
