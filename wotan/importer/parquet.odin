package importer

import "core:mem"

import w "../core"
import pq "../parquet"

// The Parquet file at `path` as a DataFrame. `columns` picks the columns to
// load, in that order (nil loads all); only their data is read from disk.
// Returns ok = false, with the reason on stderr, for files it cannot read
// faithfully (nested columns, unsupported codecs or encodings, corrupt data).
// See wotan/parquet/reader.odin for the type mapping.
parquet_load :: proc(
	path: string,
	columns: []string = nil,
	allocator: mem.Allocator = context.allocator,
) -> (
	w.DataFrame,
	bool,
) {
	return pq.read_file(path, columns, allocator)
}

// The Parquet file held in `data` (as parquet_load)
parquet_load_from_bytes :: proc(
	data: []u8,
	columns: []string = nil,
	allocator: mem.Allocator = context.allocator,
) -> (
	w.DataFrame,
	bool,
) {
	return pq.read_bytes(data, columns, allocator)
}
