package zip_min

import "core:bytes"
import "core:compress"
import "core:compress/zlib"
import "core:fmt"
import "core:mem"
import "core:os"

Local_File_Header :: struct #packed {
	signature:          u32le, // 0x04034b50
	version_needed:     u16le,
	flags:              u16le,
	compression_method: u16le,
	mod_time:           u16le,
	mod_date:           u16le,
	crc32:              u32le,
	compressed_size:    u32le,
	uncompressed_size:  u32le,
	file_name_length:   u16le,
	extra_field_length: u16le,
}

Central_Dir_Header :: struct #packed {
	signature:           u32le, // 0x02014b50
	version_made_by:     u16le,
	version_needed:      u16le,
	flags:               u16le,
	compression_method:  u16le,
	mod_time:            u16le,
	mod_date:            u16le,
	crc32:               u32le,
	compressed_size:     u32le,
	uncompressed_size:   u32le,
	file_name_length:    u16le,
	extra_field_length:  u16le,
	file_comment_length: u16le,
	disk_number_start:   u16le,
	internal_attrs:      u16le,
	external_attrs:      u32le,
	local_header_offset: u32le,
}

End_Of_Central_Dir :: struct #packed {
	signature:              u32le, // 0x06054b50
	disk_number:            u16le,
	central_dir_start_disk: u16le,
	num_entries_this_disk:  u16le,
	num_entries_total:      u16le,
	central_dir_size:       u32le,
	central_dir_offset:     u32le,
	comment_length:         u16le,
}
find_eocd :: proc(data: []u8) -> (eocd: ^End_Of_Central_Dir, ok: bool) {
	sig: u32le = 0x06054b50
	// scan last 64KB (spec limit)
	start := max(0, len(data) - 65536)
	for i := len(data) - 22; i >= start; i -= 1 {
		if len(data) - i < size_of(End_Of_Central_Dir) {
			continue
		}
		cand := (^End_Of_Central_Dir)(&data[i])
		if cand.signature == sig {
			return cand, true
		}
	}
	return nil, false
}

find_central_entry :: proc(
	data: []u8,
	eocd: ^End_Of_Central_Dir,
	name: string,
) -> (
	^Central_Dir_Header,
	bool,
) {
	sig: u32le = 0x02014b50
	off := int(eocd.central_dir_offset)
	for i in 0 ..< int(eocd.num_entries_total) {
		if off + size_of(Central_Dir_Header) > len(data) {
			return nil, false
		}
		hdr := (^Central_Dir_Header)(&data[off])
		if hdr.signature != sig {
			return nil, false
		}

		name_len := int(hdr.file_name_length)
		extra_len := int(hdr.extra_field_length)
		comment_len := int(hdr.file_comment_length)

		name_start := off + size_of(Central_Dir_Header)
		name_end := name_start + name_len
		if name_end > len(data) {
			return nil, false
		}

		entry_name := string(data[name_start:name_end])
		if entry_name == name {
			return hdr, true
		}

		off = name_end + extra_len + comment_len
	}
	return nil, false
}

read_local_and_inflate :: proc(
	data: []u8,
	hdr: ^Central_Dir_Header,
	allocator: mem.Allocator,
) -> (
	[]u8,
	bool,
) {
	local_off := int(hdr.local_header_offset)
	if local_off + size_of(Local_File_Header) > len(data) {
		return nil, false
	}

	lfh := (^Local_File_Header)(&data[local_off])
	if lfh.signature != 0x04034b50 {
		return nil, false
	}

	name_len := int(lfh.file_name_length)
	extra_len := int(lfh.extra_field_length)

	comp_start := local_off + size_of(Local_File_Header) + name_len + extra_len
	comp_end := comp_start + int(lfh.compressed_size)
	if comp_end > len(data) {
		return nil, false
	}

	comp := data[comp_start:comp_end]

	// Only support stored (0) and deflate (8)
	if lfh.compression_method == 0 {
		out := make([]u8, len(comp), allocator)
		copy(out, comp)
		return out, true
	} else if lfh.compression_method == 8 {
		buf := bytes.Buffer{}

		capacity := int(lfh.uncompressed_size)
		if capacity <= 0 {
			capacity = 0
		}
		backing := make_slice([]u8, 0, allocator)
		bytes.buffer_init(&buf, backing)

		ctx := compress.Context_Memory_Input {
			input_data = comp,
			output     = &buf,
		}
		err := zlib.inflate_raw(
			&ctx,
			expected_output_size = int(lfh.uncompressed_size),
			allocator = allocator,
		)
		if err != nil {
			bytes.buffer_destroy(&buf)
			return nil, false
		}

		out := bytes.buffer_to_bytes(&buf)
		// DO NOT destroy buf here – its backing is now logically owned by `out`
		return out, true
	}


	return nil, false
}

zip_read_file :: proc(path: string, filename: string, allocator: mem.Allocator) -> ([]u8, bool) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		return nil, false
	}
	// caller must delete(data) after use if needed
	// but we’ll copy out the payload into its own buffer
	eocd, ok := find_eocd(data)
	if !ok {

		return nil, false
	}

	cdh, ok2 := find_central_entry(data, eocd, filename)
	if !ok2 {

		return nil, false
	}

	out, ok3 := read_local_and_inflate(data, cdh, allocator)


	return out, ok3
}

zip_list :: proc(path: string, allocator: mem.Allocator) -> ([]string, bool) {
	data, err := os.read_entire_file(path, allocator)
	if err != nil {
		return nil, false
	}

	eocd, ok := find_eocd(data)
	if !ok {

		return nil, false
	}

	out := make([dynamic]string, 0, allocator)
	off := int(eocd.central_dir_offset)

	for i in 0 ..< int(eocd.num_entries_total) {
		if off + size_of(Central_Dir_Header) > len(data) {
			break
		}
		hdr := (^Central_Dir_Header)(&data[off])
		if hdr.signature != 0x02014b50 {
			break
		}

		name_len := int(hdr.file_name_length)
		extra_len := int(hdr.extra_field_length)
		comment_len := int(hdr.file_comment_length)

		name_start := off + size_of(Central_Dir_Header)
		name_end := name_start + name_len

		entry_name := string(data[name_start:name_end])
		append(&out, entry_name)

		off = name_end + extra_len + comment_len
	}

	return out[:], true
}
