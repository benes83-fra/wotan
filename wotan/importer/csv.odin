package importer

import "base:intrinsics"
import "base:runtime"
import "core:fmt"
import "core:mem"
import vmem "core:mem/virtual"
import "core:os"
import "core:strconv"
import "core:strings"

import w "../core"
import util "../util"

DEFAULT_NULL_TOKEN :: []string{"NA", "null", "None", ""}

// Null semantics (lossless with exporter.csv_save):
// - an unquoted field that is empty or a null token is null
// - in a String column a quoted field is always the literal string
//   ("" is the empty string, "NA" the text NA)
// - in typed columns quoted content is parsed like unquoted content
//
// Parsing: RFC 4180 plus a UTF-8 byte order mark, CRLF / LF / CR line ends,
// blank lines skipped, and a quote that only opens a quoted field at the
// start of the field (elsewhere it is a literal character). Rows shorter than
// the header get nulls; extra fields are ignored (with a note on stderr).
// Header names repeated are numbered as pandas does ("a", "a.1", "a.2"), and
// empty ones become "Unnamed: <position>".
//
// Types are inferred from every row unless given: Int, Float (ints and floats
// mixed), Bool (true / false), Datetime, Date, Time, Duration (text as pandas
// writes a Timedelta: "0 days 00:05:00", "-1 days +23:55:00"), else String.
//
// The reader works on the whole text at once: it cuts it into chunks at line
// ends, finds the fields of every chunk in parallel without copying them,
// infers the types per chunk and column, and converts every column of every
// chunk in parallel straight into the DataFrame (util.parallel_max_threads
// caps the threads). A quoted field with a line break that crosses a chunk
// boundary is detected and that part re-read in order, so the result never
// depends on the number of threads.

// The CSV file at path as a DataFrame; panics when the file cannot be read or
// a cell does not parse as its given type (try_csv_load returns ok = false
// instead)
csv_load :: proc(
	path: string,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
	allocator: mem.Allocator = context.allocator,
) -> w.DataFrame {
	df, err := _csv_file(path, types, sep, null_tokens, allocator, "csv_load")
	if err != "" {panic(err)}
	return df
}

// CSV text as a DataFrame (see csv_load)
csv_load_from_string :: proc(
	text: string,
	allocator: mem.Allocator = context.allocator,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> w.DataFrame {
	df, err := _csv_read(text, types, sep, null_tokens, allocator, "csv_load_from_string")
	if err != "" {panic(err)}
	return df
}

// csv_load, with ok = false (and the reason on stderr) instead of a panic
try_csv_load :: proc(
	path: string,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
	allocator: mem.Allocator = context.allocator,
) -> (
	w.DataFrame,
	bool,
) {
	df, err := _csv_file(path, types, sep, null_tokens, allocator, "try_csv_load")
	return df, _report(err)
}

// csv_load_from_string, with ok = false (and the reason on stderr) instead of
// a panic
try_csv_load_from_string :: proc(
	text: string,
	allocator: mem.Allocator = context.allocator,
	types: []w.ColumnType = nil,
	sep: u8 = ',',
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> (
	w.DataFrame,
	bool,
) {
	df, err := _csv_read(text, types, sep, null_tokens, allocator, "try_csv_load_from_string")
	return df, _report(err)
}

// Prints a loader's failure to stderr; true when there was none
@(private)
_report :: proc(err: string) -> bool {
	if err == "" {return true}
	fmt.eprintln(err)
	return false
}

@(private = "file")
_csv_file :: proc(
	path: string,
	types: []w.ColumnType,
	sep: u8,
	null_tokens: []string,
	allocator: mem.Allocator,
	label: string,
) -> (
	w.DataFrame,
	string,
) {
	// mapped rather than read: the threads fault the pages in as they go
	if mapped, merr := vmem.map_file_from_path(path, {.Read}); merr == nil && len(mapped) > 0 {
		defer vmem.unmap_file(mapped)
		return _csv_read(string(mapped), types, sep, null_tokens, allocator, label)
	}
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil {return {}, fmt.tprintf("%s: cannot read %q", label, path)}
	defer delete(data)
	return _csv_read(string(data), types, sep, null_tokens, allocator, label)
}

// ---------------------------------------------------------------------------
// Tokenizing: the fields of a chunk as spans of the text, without copying

// A field: bytes [start, start + len) of the text, relative to its chunk
@(private = "file")
Span :: struct {
	start: u32,
	len:   u32, // bit 31: the field opened with a quote
}

@(private = "file")
QUOTED :: u32(1) << 31

@(private = "file")
Chunk :: struct {
	begin, end: int, // byte range of the text
	fields:     [dynamic]Span,
	rows:       [dynamic]u32, // index of each record's first field
	in_quote:   bool, // the chunk ended inside a quoted field
	// inference and conversion
	data_rows:  int, // records that are data (the header is not)
	first:      int, // first data record (1 in the chunk holding the header)
	row_base:   int, // row of the first data record in the frame
	long_rows:  int,
}

@(private = "file")
_tokenize :: proc(text: string, ch: ^Chunk, sep: u8, last: bool) #no_bounds_check {
	clear(&ch.fields)
	clear(&ch.rows)
	i := ch.begin
	end := ch.end
	base := ch.begin
	for i < end {
		rec := len(ch.fields)
		append(&ch.rows, u32(rec))
		for { 	// fields of one record
			fs := i
			flag := u32(0)
			if i < end && text[i] == '"' {
				flag = QUOTED
				i += 1
				closed := false
				for i < end {
					c := text[i]
					if c == '"' {
						if i + 1 < end && text[i + 1] == '"' {
							i += 2
							continue
						}
						i += 1
						closed = true
						break
					}
					i += 1
				}
				if !closed && !last {
					// a quoted field runs past the chunk: the chunk is re-read
					// together with the next one
					ch.in_quote = true
					return
				}
			}
			// the rest of the field (all of an unquoted one): up to a separator
			// or a line end; quotes here are literal characters
			for i < end {
				c := text[i]
				if c == sep || c == '\n' || c == '\r' {break}
				i += 1
			}
			append(&ch.fields, Span{u32(fs - base), u32(i - fs) | flag})
			if i < end && text[i] == sep {
				i += 1
				if i == end {
					// a separator just before the end of the text: one more,
					// empty field
					append(&ch.fields, Span{u32(i - base), 0})
				}
				continue
			}
			break
		}
		// a line end (or the end of the text) closes the record
		if i < end {
			if text[i] == '\r' && i + 1 < end && text[i + 1] == '\n' {i += 2} else {i += 1}
		}
		// a blank line (one empty unquoted field) is no record
		if len(ch.fields) - rec == 1 && ch.fields[rec].len == 0 {
			pop(&ch.fields)
			pop(&ch.rows)
		}
	}
	ch.in_quote = false
}

// The text of a field: quotes removed and "" turned into " for a quoted
// field (in `scratch` when that changes the bytes); trimmed of spaces and
// tabs when `trim`
@(private = "file")
_field_text :: proc(
	text: string,
	ch: ^Chunk,
	f: Span,
	scratch: ^[dynamic]u8,
) -> (
	s: string,
	quoted: bool,
) #no_bounds_check {
	n := int(f.len & ~QUOTED)
	raw := text[ch.begin + int(f.start):][:n]
	if f.len & QUOTED == 0 {return raw, false}
	// the common case: "..." with no quote inside
	if n >= 2 && raw[n - 1] == '"' {
		inner := raw[1:n - 1]
		if strings.index_byte(inner, '"') < 0 {return inner, true}
	}
	// general case, as the reference state machine reads it: an opening
	// quote, "" for a quote, a closing quote, then literal characters
	clear(scratch)
	in_q := true
	k := 1
	for k < n {
		c := raw[k]
		if in_q && c == '"' {
			if k + 1 < n && raw[k + 1] == '"' {
				append(scratch, '"')
				k += 2
				continue
			}
			in_q = false
			k += 1
			continue
		}
		append(scratch, c)
		k += 1
	}
	return string(scratch[:]), true
}

@(private = "file")
_trim :: #force_inline proc(s: string) -> string {
	a, b := 0, len(s)
	for a < b && (s[a] == ' ' || s[a] == '\t') {a += 1}
	for b > a && (s[b - 1] == ' ' || s[b - 1] == '\t') {b -= 1}
	return s[a:b]
}

// ---------------------------------------------------------------------------
// The reader

@(private = "file")
Kind :: enum u8 {
	Int,
	Float,
	Bool,
	Datetime,
	Date,
	Time,
	Duration,
	String,
}

@(private = "file")
Kinds :: bit_set[Kind]

// A chunk's values of one column, converted while its type was inferred, in
// the chunk's own type; used as they are when the column ends up with that
// type (or Float for a chunk of Int), so most cells are parsed only once
@(private = "file")
Spec :: struct {
	kind:  Kind,
	state: u8, // 0: no value yet, 1: values in `vals`, 2: given up (mixed, text)
	vals:  []u8, // one cell per data row of the chunk
	nulls: []bool, // nil until the first null
}

@(private = "file")
_cell_type :: proc(k: Kind) -> w.ColumnType {
	switch k {
	case .Int:
		return .Int
	case .Float:
		return .Float
	case .Bool:
		return .Bool
	case .Datetime:
		return .Datetime
	case .Date:
		return .Date
	case .Time:
		return .Time
	case .Duration:
		return .Duration
	case .String:
	}
	return .String
}

// Records the trimmed, non-null text `t` of data row `lr` (of `n`) in `sp`:
// returns its kind (for inference) and keeps the value while the chunk's
// column has a single kind (Int upgrades to Float)
@(private = "file")
_classify_store :: proc(sp: ^Spec, lr, n: int, t: string) -> Kind #no_bounds_check {
	kind := Kind.String
	iv: int
	fv: f64
	bv: bool
	dv: w.Date
	tv: w.Time
	tsv: w.Datetime
	du: w.Duration
	c := t[0]
	if (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' {
		ok: bool
		if iv, ok = parse_int_strict(t); ok {
			kind = .Int
		} else if _float_syntax(t) {
			// beyond the f64 range ("1e400") it stays text, as in pandas
			if fv, ok = strconv.parse_f64(t); ok {kind = .Float}
		} else if tsv, ok = w.parse_datetime(t); ok {
			kind = .Datetime
		} else if dv, ok = w.parse_date(t); ok {
			kind = .Date
		} else if tv, ok = w.parse_time(t); ok {
			kind = .Time
		} else if fv, ok = parse_float_strict(t); ok { 	// +inf, -nan, ...
			kind = .Float
		} else if strings.contains(t, "day") {
			if du, ok = w.parse_duration(t); ok {kind = .Duration}
		}
	} else if c == 'i' || c == 'I' || c == 'n' || c == 'N' {
		if v, ok := parse_float_strict(t); ok { 	// inf, nan
			kind = .Float
			fv = v
		}
	} else if c == 't' || c == 'T' || c == 'f' || c == 'F' {
		if _ascii_fold(t, "true") || _ascii_fold(t, "false") {
			kind = .Bool
			bv = len(t) == 4
		}
	}
	if sp.state == 2 {return kind}
	if kind == .String {
		_spec_drop(sp)
		return kind
	}
	if sp.state == 0 {
		sp.kind = kind
		sp.vals = make([]u8, n * w.type_size(_cell_type(kind)))
		sp.state = 1
	} else if kind != sp.kind {
		switch {
		case sp.kind == .Int && kind == .Float:
			// earlier ints become floats (exactly as parsing their text would)
			ints := ([^]int)(raw_data(sp.vals))
			fl := make([]u8, n * 8)
			out := ([^]f64)(raw_data(fl))
			for i in 0 ..< lr {out[i] = f64(ints[i])}
			delete(sp.vals)
			sp.vals = fl
			sp.kind = .Float
		case sp.kind == .Float && kind == .Int:
			fv = f64(iv)
		case:
			_spec_drop(sp)
			return kind
		}
	}
	switch sp.kind {
	case .Int:
		([^]int)(raw_data(sp.vals))[lr] = iv
	case .Float:
		([^]f64)(raw_data(sp.vals))[lr] = fv
	case .Bool:
		([^]bool)(raw_data(sp.vals))[lr] = bv
	case .Datetime:
		([^]w.Datetime)(raw_data(sp.vals))[lr] = tsv
	case .Date:
		([^]w.Date)(raw_data(sp.vals))[lr] = dv
	case .Time:
		([^]w.Time)(raw_data(sp.vals))[lr] = tv
	case .Duration:
		([^]w.Duration)(raw_data(sp.vals))[lr] = du
	case .String:
	}
	return kind
}

@(private = "file")
_spec_null :: #force_inline proc(sp: ^Spec, lr, n: int) {
	if sp.state == 2 {return}
	if sp.nulls == nil {sp.nulls = make([]bool, n)}
	sp.nulls[lr] = true
}

@(private = "file")
_spec_drop :: proc(sp: ^Spec) {
	delete(sp.vals)
	delete(sp.nulls)
	sp^ = {
		state = 2,
	}
}

// Smallest chunk; a tiny value (-define:WOTAN_CSV_CHUNK=64) makes every few
// lines a chunk, which tests the reading of quoted fields across chunks
@(private = "file")
CSV_CHUNK_BYTES :: #config(WOTAN_CSV_CHUNK, 256 << 10)

@(private = "file")
Read_Job :: struct {
	text:        string,
	sep:         u8,
	chunks:      []Chunk,
	n_cols:      int,
	null_tokens: []string,
	infer:       bool,
	// per chunk x column (chunk major)
	kinds:       []Kinds,
	str_bytes:   []int,
	// conversion
	types:       []w.ColumnType,
	cols:        []w.Column,
	str_space:   [][]u8, // per chunk x column: the bytes reserved for strings
	specs:       []Spec, // per chunk x column: values converted while inferring
	errors:      []string, // per chunk x column: the first cell that failed
	next:        int,
	stage:       int,
}

@(private = "file")
_csv_read :: proc(
	text: string,
	types: []w.ColumnType,
	sep: u8,
	null_tokens: []string,
	allocator: mem.Allocator,
	label: string,
) -> (
	df: w.DataFrame,
	err: string,
) {
	body := text
	if strings.has_prefix(body, "\xEF\xBB\xBF") {body = body[3:]} 	// UTF-8 byte order mark
	if len(body) >= 1 << 31 {return {}, fmt.tprintf("%s: text of 2 GiB or more", label)}

	// chunks of at least 256 KiB, a few per thread, cut after a line end
	threads := util.parallel_thread_count()
	n_chunks := clamp(
		len(body) / CSV_CHUNK_BYTES,
		1,
		4 * threads if CSV_CHUNK_BYTES >= 1024 else 1 << 20,
	)
	chunks := make([dynamic]Chunk, 0, n_chunks)
	defer {
		for &c in chunks {
			delete(c.fields)
			delete(c.rows)
		}
		delete(chunks)
	}
	start := 0
	for k in 1 ..= n_chunks {
		cut := len(body) * k / n_chunks
		if k < n_chunks {
			if cut <= start {continue}
			// just past the next line end
			for cut < len(body) && body[cut] != '\n' && body[cut] != '\r' {cut += 1}
			if cut < len(body) {
				if body[cut] == '\r' &&
				   cut + 1 < len(body) &&
				   body[cut + 1] == '\n' {cut += 2} else {cut += 1}
			}
		}
		if cut > start || k == n_chunks {append(&chunks, Chunk{begin = start, end = cut})}
		start = cut
	}

	job := Read_Job {
		text        = body,
		sep         = sep,
		chunks      = chunks[:],
		null_tokens = null_tokens,
		infer       = types == nil,
	}
	util.parallel_for(len(chunks), &job, _tokenize_task, tasks = len(chunks))
	// a chunk that ended inside a quoted field is read again together with
	// the next one, until the joined chunk ends outside one
	for k := 0; k < len(chunks); k += 1 {
		for chunks[k].in_quote && k + 1 < len(chunks) {
			chunks[k].end = chunks[k + 1].end
			delete(chunks[k + 1].fields)
			delete(chunks[k + 1].rows)
			ordered_remove(&chunks, k + 1)
			_tokenize(body, &chunks[k], sep, k == len(chunks) - 1)
		}
	}
	job.chunks = chunks[:]

	// header: the first record
	hc := -1
	for c, k in chunks {
		if len(c.rows) > 0 {
			hc = k
			break
		}
	}
	if hc < 0 {return {}, fmt.tprintf("%s: empty input", label)}
	h := &chunks[hc]
	h_end := len(h.fields) if len(h.rows) == 1 else int(h.rows[1])
	n_cols := h_end - int(h.rows[0])
	if types != nil && len(types) != n_cols {
		return {}, fmt.tprintf("%s: %d types for %d header fields", label, len(types), n_cols)
	}
	job.n_cols = n_cols
	rows := 0
	for &c, k in chunks {
		c.first = 1 if k == hc else 0
		c.data_rows = max(len(c.rows) - c.first, 0)
		c.row_base = rows
		rows += c.data_rows
	}

	// names: repeats numbered, blanks named, as pandas does
	names := make([]string, n_cols, context.temp_allocator)
	scratch := make([dynamic]u8, context.temp_allocator)
	for j in 0 ..< n_cols {
		s, _ := _field_text(body, h, h.fields[int(h.rows[0]) + j], &scratch)
		names[j] = strings.clone(s, context.temp_allocator)
	}
	_unique_names(names)

	// types and string sizes, per chunk and column
	job.kinds = make([]Kinds, len(chunks) * n_cols)
	job.str_bytes = make([]int, len(chunks) * n_cols)
	job.specs = make([]Spec, len(chunks) * n_cols)
	defer {
		for &sp in job.specs {
			delete(sp.vals, runtime.heap_allocator())
			delete(sp.nulls, runtime.heap_allocator())
		}
		delete(job.specs)
	}
	defer delete(job.kinds)
	defer delete(job.str_bytes)
	job.next = 0
	util.parallel_for(len(chunks), &job, _scan_task, tasks = len(chunks))
	out_types := make([]w.ColumnType, n_cols, context.temp_allocator)
	for j in 0 ..< n_cols {
		if types != nil {
			out_types[j] = types[j]
			continue
		}
		seen: Kinds
		for k in 0 ..< len(chunks) {seen += job.kinds[k * n_cols + j]}
		out_types[j] = _type_of(seen)
	}
	job.types = out_types

	// the columns, and their string bytes reserved chunk by chunk
	cols := make([]w.Column, n_cols)
	defer delete(cols)
	built := false
	defer if !built {
		for &c in cols {w.destroy_column(&c)}
	}
	job.str_space = make([][]u8, len(chunks) * n_cols)
	defer delete(job.str_space)
	for j in 0 ..< n_cols {
		cols[j] = w.column_new("", out_types[j], max(rows, 1), allocator)
		w.column_own_name(&cols[j], names[j])
		if out_types[j] != .String {continue}
		total := 0
		for k in 0 ..< len(chunks) {total += job.str_bytes[k * n_cols + j]}
		if total == 0 {continue}
		if cols[j].arena.curr_block == nil {
			if vmem.arena_init_growing(&cols[j].arena) !=
			   nil {return {}, fmt.tprintf("%s: out of memory", label)}
		}
		block, aerr := vmem.arena_alloc(&cols[j].arena, uint(total), 1)
		if aerr != nil {return {}, fmt.tprintf("%s: out of memory", label)}
		off := 0
		for k in 0 ..< len(chunks) {
			nb := job.str_bytes[k * n_cols + j]
			job.str_space[k * n_cols + j] = block[off:off + nb]
			off += nb
		}
	}
	job.cols = cols
	job.errors = make([]string, len(chunks) * n_cols)
	defer delete(job.errors)
	job.next = 0
	util.parallel_for(len(chunks) * n_cols, &job, _convert_task, tasks = len(chunks) * n_cols)
	first_err := ""
	for e in job.errors {
		if e != "" && first_err == "" {first_err = fmt.tprintf("%s: %s", label, e)}
		if e != "" {delete(e, runtime.heap_allocator())}
	}
	if first_err != "" {return {}, first_err}

	long_rows := 0
	for c in chunks {long_rows += c.long_rows}
	if long_rows > 0 {
		fmt.eprintf(
			"%s: %d row(s) had more than %d fields; extra fields ignored\n",
			label,
			long_rows,
			n_cols,
		)
	}

	df = w.dataframe_new(allocator)
	for &c in cols {
		c.len = rows
		w.add_column(&df, c)
	}
	df.rows = rows
	built = true
	return df, ""
}

// Repeated names get ".1", ".2", ... (skipping names already taken); empty
// names become "Unnamed: <position>"
@(private)
_unique_names :: proc(names: []string) {
	taken := make(map[string]bool, len(names), context.temp_allocator)
	for &n, j in names {
		if n == "" {n = fmt.tprintf("Unnamed: %d", j)}
	}
	for n in names {taken[n] = true}
	seen := make(map[string]bool, len(names), context.temp_allocator)
	for &n in names {
		if !(n in seen) {
			seen[n] = true
			continue
		}
		base := n
		for k := 1;; k += 1 {
			cand := fmt.tprintf("%s.%d", base, k)
			if !(cand in taken) {
				n = cand
				taken[cand] = true
				seen[cand] = true
				break
			}
		}
	}
}

@(private = "file")
_tokenize_task :: proc(data: rawptr, begin, end: int) {
	job := (^Read_Job)(data)
	context.allocator = runtime.heap_allocator()
	for {
		k := intrinsics.atomic_add(&job.next, 1)
		if k >= len(job.chunks) {break}
		_tokenize(job.text, &job.chunks[k], job.sep, k == len(job.chunks) - 1)
	}
}

// Whether a trimmed, unquoted field is one of the null tokens (a token in
// quotes counts too, as before)
@(private = "file")
_is_null_token :: #force_inline proc(s: string, tokens: []string) -> bool {
	t := s
	if len(t) >= 2 && t[0] == '"' && t[len(t) - 1] == '"' {t = unquote_and_trim(t)}
	for tok in tokens {
		if t == tok {return true}
	}
	return false
}

// The column type for the kinds seen, as infer_column_type decides it
@(private = "file")
_type_of :: proc(seen: Kinds) -> w.ColumnType {
	numeric := .Int in seen || .Float in seen
	groups :=
		int(numeric) +
		int(.Bool in seen) +
		int(.Datetime in seen) +
		int(.Date in seen) +
		int(.Time in seen) +
		int(.Duration in seen) +
		int(.String in seen)
	switch {
	case groups != 1:
		return .String
	case .Float in seen:
		return .Float
	case .Int in seen:
		return .Int
	case .Bool in seen:
		return .Bool
	case .Datetime in seen:
		return .Datetime
	case .Date in seen:
		return .Date
	case .Time in seen:
		return .Time
	case .Duration in seen:
		return .Duration
	}
	return .String
}

// Per chunk: the kinds in every column (when inferring) and the string bytes
@(private = "file")
_scan_task :: proc(data: rawptr, begin, end: int) {
	job := (^Read_Job)(data)
	context.allocator = runtime.heap_allocator()
	scratch := make([dynamic]u8)
	defer delete(scratch)
	nc := job.n_cols
	for {
		k := intrinsics.atomic_add(&job.next, 1)
		if k >= len(job.chunks) {break}
		ch := &job.chunks[k]
		// local tallies, written back once: neighbouring chunks' slots share
		// cache lines
		kinds := make([]Kinds, nc)
		bytes := make([]int, nc)
		specs := job.specs[k * nc:(k + 1) * nc]
		if !job.infer {
			for &sp in specs {sp.state = 2}
		}
		n := ch.data_rows
		long_rows := 0
		for r in ch.first ..< len(ch.rows) {
			lr := r - ch.first
			f0 := int(ch.rows[r])
			f1 := len(ch.fields) if r + 1 == len(ch.rows) else int(ch.rows[r + 1])
			if f1 - f0 > nc {long_rows += 1}
			for j in 0 ..< nc {
				if j >= f1 - f0 { 	// short row
					_spec_null(&specs[j], lr, n)
					continue
				}
				f := ch.fields[f0 + j]
				bytes[j] += int(f.len & ~QUOTED)
				// text makes the column a String column whatever follows
				if !job.infer || .String in kinds[j] {continue}
				s, _ := _field_text(job.text, ch, f, &scratch)
				t := _trim(s)
				if _is_null_token(t, job.null_tokens) {
					_spec_null(&specs[j], lr, n)
					continue
				}
				if t == "NA" || t == "null" || t == "None" || t == "NaN" {
					// no kind (as infer_column_type), but not a null token
					// either: left to the conversion
					if specs[j].state != 2 {_spec_drop(&specs[j])}
					continue
				}
				kinds[j] += {_classify_store(&specs[j], lr, n, t)}
			}
		}
		copy(job.kinds[k * nc:(k + 1) * nc], kinds)
		copy(job.str_bytes[k * nc:(k + 1) * nc], bytes)
		ch.long_rows = long_rows
		delete(kinds)
		delete(bytes)
	}
}

// Per chunk and column: the cells, converted into the column
@(private = "file")
_convert_task :: proc(data: rawptr, begin, end: int) {
	job := (^Read_Job)(data)
	context.allocator = runtime.heap_allocator()
	scratch := make([dynamic]u8)
	defer delete(scratch)
	nc := job.n_cols
	for {
		t := intrinsics.atomic_add(&job.next, 1)
		if t >= len(job.chunks) * nc {break}
		k, j := t / nc, t % nc
		ch := &job.chunks[k]
		col := &job.cols[j]
		typ := job.types[j]
		space := job.str_space[t]
		used := 0
		sp := &job.specs[t]
		if typ != .String && sp.state != 2 {
			n := ch.data_rows
			base := ch.row_base
			if sp.state == 0 { 	// every cell null
				for i in 0 ..< n {col.nulls[base + i] = true}
				continue
			}
			if _cell_type(sp.kind) == typ || (sp.kind == .Int && typ == .Float) {
				if sp.kind == .Int && typ == .Float {
					ints := ([^]int)(raw_data(sp.vals))
					out := ([^]f64)(col.data)
					for i in 0 ..< n {out[base + i] = f64(ints[i])}
				} else {
					sz := w.type_size(typ)
					intrinsics.mem_copy_non_overlapping(
						rawptr(uintptr(col.data) + uintptr(base * sz)),
						raw_data(sp.vals),
						n * sz,
					)
				}
				if sp.nulls != nil {copy(col.nulls[base:base + n], sp.nulls)}
				continue
			}
		}
		for r in ch.first ..< len(ch.rows) {
			row := ch.row_base + r - ch.first
			f0 := int(ch.rows[r])
			f1 := len(ch.fields) if r + 1 == len(ch.rows) else int(ch.rows[r + 1])
			if j >= f1 - f0 { 	// short row: null
				col.nulls[row] = true
				continue
			}
			s, quoted := _field_text(job.text, ch, ch.fields[f0 + j], &scratch)
			if typ == .String {
				if !quoted {
					s = _trim(s)
					if _is_null_token(s, job.null_tokens) {
						col.nulls[row] = true
						continue
					}
				}
				copy(space[used:], s)
				([^]string)(col.data)[row] = string(space[used:used + len(s)])
				used += len(s)
				continue
			}
			field := _trim(s)
			if _is_null_token(field, job.null_tokens) {
				col.nulls[row] = true
				continue
			}
			ok := true
			#partial switch typ {
			case .Int:
				([^]int)(col.data)[row], ok = parse_int_strict(field)
			case .Float:
				([^]f64)(col.data)[row], ok = parse_float_strict(field)
			case .Bool:
				truthy := _ascii_fold(field, "true") || field == "1"
				ok = truthy || _ascii_fold(field, "false") || field == "0"
				([^]bool)(col.data)[row] = truthy
			case .Date:
				([^]w.Date)(col.data)[row], ok = w.parse_date(field)
			case .Time:
				([^]w.Time)(col.data)[row], ok = w.parse_time(field)
			case .Datetime:
				([^]w.Datetime)(col.data)[row], ok = w.parse_datetime(field)
			case .Duration:
				([^]w.Duration)(col.data)[row], ok = w.parse_duration(field)
			}
			if !ok && (field == "NA" || field == "null" || field == "None" || field == "NaN") {
				// texts that inference passes over: missing values (NaN
				// parses as NaN in a Float column)
				col.nulls[row] = true
				ok = true
			}
			if !ok {
				// inferred types fit every cell, so this only happens with
				// explicit types
				job.errors[t] = fmt.aprintf(
					"row %d, column %q: cannot parse '%s' as %v",
					row + 1,
					col.name,
					field,
					typ,
				) // heap (this task's context); freed by _csv_read
				break
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers kept for other importers and callers

// Types inferred from every data row of parsed records (fields trimmed; null
// tokens skipped)
infer_col_types :: proc(
	records: [][]string,
	null_tokens: []string = DEFAULT_NULL_TOKEN,
) -> []w.ColumnType {
	header := records[0]
	col_count := len(header)
	samples := make([][dynamic]string, col_count)
	defer {
		for i in 0 ..< col_count {delete(samples[i])}
		delete(samples)
	}
	for row_i in 1 ..< len(records) {
		fields := records[row_i]
		for col_i in 0 ..< min(col_count, len(fields)) {
			if is_null_field(fields[col_i], null_tokens) {continue}
			append(&samples[col_i], strings.trim(fields[col_i], " \t"))
		}
	}
	types := make([]w.ColumnType, col_count, context.temp_allocator)
	for i in 0 ..< col_count {
		types[i] = infer_column_type(samples[i][:])
	}
	return types
}

// The records of CSV text as strings (the fields of each record in order;
// quotes removed). Free with csv_free.
parse_csv_records :: proc(text: string, seperator: u8 = ',') -> [][]string {
	body := text
	if strings.has_prefix(body, "\xEF\xBB\xBF") {body = body[3:]}
	ch := Chunk {
		begin = 0,
		end   = len(body),
	}
	defer {
		delete(ch.fields)
		delete(ch.rows)
	}
	_tokenize(body, &ch, seperator, true)
	scratch := make([dynamic]u8)
	defer delete(scratch)
	out := make([][]string, len(ch.rows))
	for r in 0 ..< len(ch.rows) {
		f0 := int(ch.rows[r])
		f1 := len(ch.fields) if r + 1 == len(ch.rows) else int(ch.rows[r + 1])
		rec := make([]string, f1 - f0)
		for j in 0 ..< f1 - f0 {
			s, _ := _field_text(body, &ch, ch.fields[f0 + j], &scratch)
			rec[j] = strings.clone(s, context.temp_allocator) if s != "" else ""
		}
		out[r] = rec
	}
	return out
}

csv_free :: proc(records: [][]string) {
	for record in records {
		if record != nil {
			for &s in record {
				if s != "" {
					delete(s, context.temp_allocator)
				}
			}
			delete(record)
		}
	}
	if records != nil {
		delete(records)
	}
}

// Helper to trim optional surrounding whitespace and quotes, and unescape double quotes.
// Use this when you want to treat quoted empty string as empty string and remove outer quotes.
unquote_and_trim :: proc(str: string) -> string {
	s := strings.trim(str, " \t")
	if len(s) >= 2 && s[0] == '"' && s[len(s) - 1] == '"' {
		// remove surrounding quotes and unescape double quotes
		inner := s[1:len(s) - 1]
		out_bytes := make([dynamic]u8, context.temp_allocator)
		i := 0
		for i < len(inner) {
			if inner[i] == '"' && i + 1 < len(inner) && inner[i + 1] == '"' {
				append(&out_bytes, u8('"'))
				i += 2
			} else {
				append(&out_bytes, u8(inner[i]))
				i += 1
			}
		}
		return string(out_bytes[:])
	}
	return s
}

is_token_equal :: proc(a: string, b: string) -> bool {
	return a == b
}

// True if the (unquoted) field text is a null token after trimming
is_null_field :: proc(field: string, null_tokens: []string) -> bool {
	trimmed := strings.trim(field, " \t")
	if len(trimmed) >= 2 && trimmed[0] == '"' && trimmed[len(trimmed) - 1] == '"' {
		trimmed = unquote_and_trim(trimmed)
	}
	for t in null_tokens {
		if is_token_equal(trimmed, t) {
			return true
		}
	}
	return false
}

// parse_float_strict's syntax check without computing the value
@(private = "file")
_float_syntax :: proc(s: string) -> bool {
	t := s
	if len(t) > 0 && (t[0] == '+' || t[0] == '-') {t = t[1:]}
	i, digits := 0, 0
	for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1; digits += 1}
	if i < len(t) && t[i] == '.' {
		i += 1
		for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1; digits += 1}
	}
	if digits == 0 {return false}
	if i < len(t) && (t[i] == 'e' || t[i] == 'E') {
		i += 1
		if i < len(t) && (t[i] == '+' || t[i] == '-') {i += 1}
		e := i
		for i < len(t) && t[i] >= '0' && t[i] <= '9' {i += 1}
		if i == e {return false}
	}
	return i == len(t)
}

// ASCII case-insensitive equality with a lower-case word
@(private = "file")
_ascii_fold :: #force_inline proc(s, lower: string) -> bool {
	if len(s) != len(lower) {return false}
	for i in 0 ..< len(s) {
		if s[i] | 0x20 != lower[i] {return false}
	}
	return true
}
