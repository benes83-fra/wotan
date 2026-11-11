package net

import w "../core"
import "../importer"
import "base:runtime"
import "core:c"
import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import curl "vendor:curl"

ResponseBuffer :: struct {
	data: [dynamic]u8,
}

// Callback must use c_size_t and match the C signature exactly
write_cb :: proc "c" (ptr: [^]u8, size: uint, nmemb: uint, userdata: rawptr) -> uint {
	context = runtime.default_context()
	total := int(size * nmemb)
	buf := cast(^ResponseBuffer)userdata

	// Efficiently append the entire chunk
	append_elems(&buf.data, ..ptr[:total])

	return c.size_t(total)
}

// The body of a GET request; ok is false only when the request fails (any HTTP
// status counts as a response)
http_get :: proc(url: string, allocator: mem.Allocator = context.allocator) -> (string, bool) {
	body, _, ok := _http_get_status(url, allocator)
	return body, ok
}

// Saves the body of a GET request to path. ok is false when the request fails, the
// server answers with an HTTP status outside 200 - 299 or the file cannot be written
download_file :: proc(url, path: string) -> bool {
	body, status, ok := _http_get_status(url, context.temp_allocator)
	defer delete(body, context.temp_allocator)
	if !ok || status < 200 || status > 299 {return false}
	return os.write_entire_file(path, transmute([]u8)body) == nil
}

@(private = "file")
_http_get_status :: proc(
	url: string,
	allocator: mem.Allocator,
) -> (
	body: string,
	status: int,
	ok: bool,
) {
	curl_handle := curl.easy_init()
	if curl_handle == nil {
		return "", 0, false
	}
	defer curl.easy_cleanup(curl_handle)

	rb := ResponseBuffer {
		data = make([dynamic]u8, allocator),
	}
	// Note: We don't delete rb.data here because we return its content as a string

	// libcurl needs a null-terminated C string
	c_url := strings.clone_to_cstring(url, context.temp_allocator)

	curl.easy_setopt(curl_handle, .URL, c_url)
	curl.easy_setopt(curl_handle, .WRITEFUNCTION, write_cb)
	curl.easy_setopt(curl_handle, .WRITEDATA, &rb)

	// Follow redirects (common for URLs)
	curl.easy_setopt(curl_handle, .FOLLOWLOCATION, i64(1))
	curl.easy_setopt(curl_handle, curl.option.USERAGENT, "Mozilla/5.0")
	curl.easy_setopt(curl_handle, curl.option.COOKIEFILE, "")
	curl.easy_setopt(curl_handle, curl.option.COOKIEJAR, "")


	res := curl.easy_perform(curl_handle)
	if res != .E_OK {
		delete(rb.data)
		return "", 0, false
	}
	code: c.long
	curl.easy_getinfo(curl_handle, .RESPONSE_CODE, &code)

	return string(rb.data[:]), int(code), true
}

read_csv_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	// Use temp_allocator for the raw response text to avoid leaks
	text, ok := http_get(url, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Failed to GET %s", url))
	}

	// Now we pass the string to the importer
	return importer.csv_load_from_string(text, allocator)
}
read_html_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	// Fetch HTML text using your existing HTTP client
	text, ok := http_get(url, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Failed to GET %s", url))
	}

	// Pass HTML to your importer
	return importer.html_load_from_string(text, allocator)
}
read_table_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	text, ok := http_get(url, context.temp_allocator)
	if !ok {
		panic(fmt.tprintf("Failed to GET %s", url))
	}

	if strings.contains(text, "<table") {
		return importer.html_load_from_string(text, allocator)
	}

	return importer.csv_load_from_string(text, allocator)
}


http_get_with_cookie :: proc(
	url: string,
	cookie: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	string,
	bool,
) {

	curl_handle := curl.easy_init()
	if curl_handle == nil {
		return "", false
	}
	defer curl.easy_cleanup(curl_handle)

	rb := ResponseBuffer {
		data = make([dynamic]u8, allocator),
	}

	// URL
	c_url := strings.clone_to_cstring(url, context.temp_allocator)
	curl.easy_setopt(curl_handle, curl.option.URL, c_url)

	// Write callback
	curl.easy_setopt(curl_handle, curl.option.WRITEFUNCTION, write_cb)
	curl.easy_setopt(curl_handle, curl.option.WRITEDATA, &rb)

	// Follow redirects
	curl.easy_setopt(curl_handle, curl.option.FOLLOWLOCATION, i64(1))

	// Browser-like headers
	curl.easy_setopt(curl_handle, curl.option.USERAGENT, "Mozilla/5.0")

	// Enable cookie engine
	curl.easy_setopt(curl_handle, curl.option.COOKIEFILE, "")
	curl.easy_setopt(curl_handle, curl.option.COOKIEJAR, "")

	// Attach cookie
	c_cookie := strings.clone_to_cstring(cookie, context.temp_allocator)
	curl.easy_setopt(curl_handle, curl.option.COOKIE, c_cookie)

	// Perform request
	res := curl.easy_perform(curl_handle)
	if res != .E_OK {
		delete(rb.data)
		return "", false
	}

	return string(rb.data[:]), true
}
