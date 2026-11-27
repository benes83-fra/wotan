package exporter

import w "../core"
import pq "../parquet"

// Codec of the pages: .Snappy (the default, as in pandas), .Zstd (smaller,
// a little slower), .Gzip (a little larger than .Zstd and several times
// slower), .Lz4 (written as LZ4_RAW) or .None. .Zstd and .Gzip take a
// `compression_level` (see parquet_save)
Parquet_Compression :: pq.Compression

// Writes `df` as a Parquet file that pandas, pyarrow, Spark, DuckDB and
// Polars read with the same types (see wotan/parquet/writer.odin for the
// mapping). `row_group_rows` rows go into each row group; `dictionary`
// dictionary-encodes String columns while their dictionary stays under
// 1 MiB. `compression_level` is .Zstd's level, 1 (fastest) to 12 (smallest),
// or .Gzip's, 1 to 9; 0 takes the codec's default (3 for .Zstd, 6 for .Gzip)
// and levels out of range are clamped. Returns false, with the reason on
// stderr, if the file cannot be written.
parquet_save :: proc(
	df: ^w.DataFrame,
	path: string,
	compression := Parquet_Compression.Snappy,
	row_group_rows := 1 << 20,
	dictionary := true,
	compression_level := 0,
) -> bool {
	return pq.write_file(df, path, compression, row_group_rows, dictionary, compression_level)
}

// The bytes of the Parquet file parquet_save would write; free them with
// delete
parquet_bytes :: proc(
	df: ^w.DataFrame,
	compression := Parquet_Compression.Snappy,
	row_group_rows := 1 << 20,
	dictionary := true,
	compression_level := 0,
	allocator := context.allocator,
) -> (
	[]u8,
	bool,
) {
	return pq.write_bytes(
		df,
		compression,
		row_group_rows,
		dictionary,
		compression_level,
		allocator,
	)
}
