package zip_min

import "core:bytes"
import "core:compress"
import "core:compress/zlib"
import "core:fmt"
import "core:mem"
import "core:os"

// ------------------------------------------------------------
// Helpers: little-endian decoding from []u8 (ARM-safe)
// ------------------------------------------------------------

read_u16_le :: proc(b: []u8, off: int) -> u16 {
	return u16(b[off + 0]) | u16(b[off + 1]) << 8
}

read_u32_le :: proc(b: []u8, off: int) -> u32 {
	return u32(b[off + 0]) | u32(b[off + 1]) << 8 | u32(b[off + 2]) << 16 | u32(b[off + 3]) << 24
}

// ------------------------------------------------------------
// Struct sizes (for bounds checks) – match ZIP spec
// ------------------------------------------------------------

LOCAL_FILE_HEADER_SIZE :: 30 // fixed part
CENTRAL_DIR_HEADER_SIZE :: 46 // fixed part
END_OF_CENTRAL_DIR_SIZE :: 22 // fixed part

// ------------------------------------------------------------
// EOCD parsing
// ------------------------------------------------------------

find_eocd :: proc(data: []u8) -> (off: int, ok: bool) {
	// EOCD signature: 0x06054b50 (little endian: 50 4b 05 06)
	sig0: u8 = 0x50
	sig1: u8 = 0x4b
	sig2: u8 = 0x05
	sig3: u8 = 0x06

	if len(data) < END_OF_CENTRAL_DIR_SIZE {
		return -1, false
	}

	start := max(0, len(data) - 65536)
	for i := len(data) - END_OF_CENTRAL_DIR_SIZE; i >= start; i -= 1 {
		if data[i + 0] == sig0 &&
		   data[i + 1] == sig1 &&
		   data[i + 2] == sig2 &&
		   data[i + 3] == sig3 {
			return i, true
		}
	}

	return -1, false
}

// ------------------------------------------------------------
// Central directory entry parsing
// ------------------------------------------------------------

find_central_entry :: proc(data: []u8, eocd_off: int, name: string) -> (hdr_off: int, ok: bool) {
	if eocd_off + END_OF_CENTRAL_DIR_SIZE > len(data) {
		return -1, false
	}

	// EOCD layout (little endian):
	//  0: signature (4)
	//  4: disk_number (2)
	//  6: central_dir_start_disk (2)
	//  8: num_entries_this_disk (2)
	// 10: num_entries_total (2)
	// 12: central_dir_size (4)
	// 16: central_dir_offset (4)
	// 20: comment_length (2)

	num_entries_total := int(read_u16_le(data, eocd_off + 10))
	central_dir_offset := int(read_u32_le(data, eocd_off + 16))

	off := central_dir_offset
	for i := 0; i < num_entries_total; i += 1 {
		if off + CENTRAL_DIR_HEADER_SIZE > len(data) {
			return -1, false
		}

		// Central dir header signature: 0x02014b50
		if read_u32_le(data, off + 0) != u32(0x02014b50) {
			return -1, false
		}

		file_name_length := int(read_u16_le(data, off + 28))
		extra_field_length := int(read_u16_le(data, off + 30))
		file_comment_length := int(read_u16_le(data, off + 32))

		name_start := off + CENTRAL_DIR_HEADER_SIZE
		name_end := name_start + file_name_length
		if name_end > len(data) {
			return -1, false
		}

		entry_name := string(data[name_start:name_end])
		if entry_name == name {
			return off, true
		}

		off = name_end + extra_field_length + file_comment_length
	}

	return -1, false
}

// ------------------------------------------------------------
// Local header + payload
// ------------------------------------------------------------

read_local_and_inflate :: proc(
	data: []u8,
	cdh_off: int,
	allocator: mem.Allocator,
) -> (
	[]u8,
	bool,
) {
	if cdh_off + CENTRAL_DIR_HEADER_SIZE > len(data) {
		return nil, false
	}

	// From central dir header:
	compression_method := read_u16_le(data, cdh_off + 10)
	compressed_size := int(read_u32_le(data, cdh_off + 20))
	uncompressed_size := int(read_u32_le(data, cdh_off + 24))
	local_header_off := int(read_u32_le(data, cdh_off + 42))

	// Local file header layout:
	//  0: signature (4) = 0x04034b50
	//  4: version_needed (2)
	//  6: flags (2)
	//  8: compression_method (2)
	// 10: mod_time (2)
	// 12: mod_date (2)
	// 14: crc32 (4)
	// 18: compressed_size (4)
	// 22: uncompressed_size (4)
	// 26: file_name_length (2)
	// 28: extra_field_length (2)

	if local_header_off + LOCAL_FILE_HEADER_SIZE > len(data) {
		return nil, false
	}
	if read_u32_le(data, local_header_off + 0) != u32(0x04034b50) {
		return nil, false
	}

	file_name_length := int(read_u16_le(data, local_header_off + 26))
	extra_field_len := int(read_u16_le(data, local_header_off + 28))

	comp_start := local_header_off + LOCAL_FILE_HEADER_SIZE + file_name_length + extra_field_len
	comp_end := comp_start + compressed_size
	if comp_end > len(data) {
		return nil, false
	}

	comp := data[comp_start:comp_end]

	// stored
	if compression_method == 0 {
		out := make([]u8, len(comp), allocator)
		copy(out, comp)
		return out, true
	}

	// deflate
	if compression_method == 8 {
		buf := bytes.Buffer{}
		backing := make_slice([]u8, 0, allocator)
		bytes.buffer_init(&buf, backing)

		ctx := compress.Context_Memory_Input {
			input_data = comp,
			output     = &buf,
		}
		err := zlib.inflate_raw(
			&ctx,
			expected_output_size = uncompressed_size,
			allocator = allocator,
		)
		if err != nil {
			bytes.buffer_destroy(&buf)
			return nil, false
		}

		out := bytes.buffer_to_bytes(&buf)
		// do not destroy buf – backing is now owned by `out`
		return out, true
	}

	// unsupported compression
	return nil, false
}

// ------------------------------------------------------------
// Public API
// ------------------------------------------------------------

zip_read_file :: proc(path: string, filename: string, allocator: mem.Allocator) -> ([]u8, bool) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		return nil, false
	}
	// NOTE: we never cast data to header structs; everything is parsed manually.

	eocd_off, ok := find_eocd(data)
	if !ok {
		fmt.println("zip_read_file: EOCD not found")
		return nil, false
	}

	cdh_off, ok2 := find_central_entry(data, eocd_off, filename)
	if !ok2 {
		fmt.println("zip_read_file: central entry not found for", filename)
		return nil, false
	}

	out, ok3 := read_local_and_inflate(data, cdh_off, allocator)
	return out, ok3
}

zip_list :: proc(path: string, allocator: mem.Allocator) -> ([]string, bool) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		return nil, false
	}

	eocd_off, ok := find_eocd(data)
	if !ok {
		return nil, false
	}

	num_entries_total := int(read_u16_le(data, eocd_off + 10))
	central_dir_offset := int(read_u32_le(data, eocd_off + 16))

	out := make([dynamic]string, 0, allocator)
	off := central_dir_offset

	for i := 0; i < num_entries_total; i += 1 {
		if off + CENTRAL_DIR_HEADER_SIZE > len(data) {
			break
		}
		if read_u32_le(data, off + 0) != u32(0x02014b50) {
			break
		}

		file_name_length := int(read_u16_le(data, off + 28))
		extra_field_length := int(read_u16_le(data, off + 30))
		file_comment_length := int(read_u16_le(data, off + 32))

		name_start := off + CENTRAL_DIR_HEADER_SIZE
		name_end := name_start + file_name_length
		if name_end > len(data) {
			break
		}

		entry_name := string(data[name_start:name_end])
		append(&out, entry_name)

		off = name_end + extra_field_length + file_comment_length
	}

	return out[:], true
}
