package importer

import "core:mem"

import w "../core"
import pq "../parquet"

// Row filter of parquet_load: keeps the rows whose `column` cell lies between
// `lower` and `upper` (nil leaves an end open; `inclusive` says which ends
// belong to the range, both by default). Missing cells and NaN never match.
Parquet_Filter :: pq.Filter
// A filter bound or a statistics value: int, f64, bool, string, Date, Time or
// Datetime
Parquet_Value :: pq.Value
// What parquet_info returns: rows, columns (name, DataFrame type, stored
// type) and per-row-group statistics (null count, min, max)
Parquet_Info :: pq.File_Info
Parquet_Column_Info :: pq.Column_Info
Parquet_Row_Group_Info :: pq.Row_Group_Info
Parquet_Column_Stats :: pq.Column_Stats

// The Parquet file at `path` as a DataFrame. `columns` picks the columns to
// load, in that order (nil loads all); only their data is read from disk.
// `filters` keeps the rows that match every filter; row groups whose
// statistics show no match are skipped without being read. A filter column
// need not be among `columns`.
// Returns ok = false, with the reason on stderr, for files it cannot read
// faithfully (nested columns, unsupported codecs or encodings, corrupt data)
// and for a filter on a missing column or with bounds of the wrong type.
// See wotan/parquet/reader.odin for the type mapping.
parquet_load :: proc(
	path: string,
	columns: []string = nil,
	allocator: mem.Allocator = context.allocator,
	filters: []Parquet_Filter = nil,
) -> (
	w.DataFrame,
	bool,
) {
	return pq.read_file(path, columns, allocator, filters)
}

// The Parquet file held in `data` (as parquet_load)
parquet_load_from_bytes :: proc(
	data: []u8,
	columns: []string = nil,
	allocator: mem.Allocator = context.allocator,
	filters: []Parquet_Filter = nil,
) -> (
	w.DataFrame,
	bool,
) {
	return pq.read_bytes(data, columns, allocator, filters)
}

// Rows, columns and per-row-group statistics of the Parquet file at `path`,
// from its footer alone (no data is read). Free with destroy_parquet_info.
// See wotan/parquet/info.odin for when min / max are nil.
parquet_info :: proc(path: string, allocator := context.allocator) -> (Parquet_Info, bool) {
	return pq.info_file(path, allocator)
}

// parquet_info for a Parquet file held in `data`
parquet_info_from_bytes :: proc(
	data: []u8,
	allocator := context.allocator,
) -> (
	Parquet_Info,
	bool,
) {
	return pq.info_bytes(data, allocator)
}

destroy_parquet_info :: proc(info: ^Parquet_Info) {
	pq.destroy_info(info)
}

// Prints the row count, the row groups and a line per column: DataFrame type,
// stored type, null count and the range of values over the file
print_parquet_info :: proc(info: ^Parquet_Info) {
	pq.print_info(info)
}
