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

// GET body of url when the request succeeds with an HTTP status in 200 - 299
@(private = "file")
_get_ok_body :: proc(url: string) -> (string, bool) {
	body, status, ok := _http_get_status(url, context.temp_allocator)
	return body, ok && status >= 200 && status <= 299
}

// The CSV at url as a DataFrame (see importer.csv_load_from_string). ok is false
// when the request fails or the server answers with a status outside 200 - 299;
// the DataFrame is then empty
read_csv_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	text, got := _get_ok_body(url)
	if !got {return w.dataframe_new(), false}
	return importer.csv_load_from_string(text, allocator), true
}

// The first <table> of the HTML page at url as a DataFrame (see
// importer.html_load_from_string); ok as for read_csv_from_url
read_html_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	text, got := _get_ok_body(url)
	if !got {return w.dataframe_new(), false}
	return importer.html_load_from_string(text, allocator), true
}

// read_html_from_url when the body contains a <table>, read_csv_from_url otherwise
read_table_from_url :: proc(
	url: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	df: w.DataFrame,
	ok: bool,
) {
	text, got := _get_ok_body(url)
	if !got {return w.dataframe_new(), false}
	if strings.contains(text, "<table") {
		return importer.html_load_from_string(text, allocator), true
	}
	return importer.csv_load_from_string(text, allocator), true
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
