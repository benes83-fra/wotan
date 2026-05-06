package exporter

import "core:bytes"
import "core:hash"
import "core:mem"
import "core:os"

ZipEntry :: struct {
	name: string,
	data: []u8,
}

zip_write :: proc(path: string, entries: []ZipEntry, allocator: mem.Allocator) -> bool {
	// Build ZIP in memory
	backing := make([]u8, 0, allocator)
	buf := bytes.Buffer{}
	bytes.buffer_init(&buf, backing)
	defer bytes.buffer_destroy(&buf)

	local_offsets := make([]int, len(entries), allocator)

	write_u16_le := proc(b: ^bytes.Buffer, x: u16) {
		bytes.buffer_write_byte(b, u8(x))
		bytes.buffer_write_byte(b, u8(x >> 8))
	}
	write_u32_le := proc(b: ^bytes.Buffer, x: u32) {
		bytes.buffer_write_byte(b, u8(x))
		bytes.buffer_write_byte(b, u8(x >> 8))
		bytes.buffer_write_byte(b, u8(x >> 16))
		bytes.buffer_write_byte(b, u8(x >> 24))
	}

	// ---- Local file headers ----
	for i in 0 ..< len(entries) {
		e := entries[i]

		name_bytes := transmute([]u8)e.name
		data_bytes := e.data

		local_offsets[i] = bytes.buffer_length(&buf)

		write_u32_le(&buf, 0x04034b50) // local header sig
		write_u16_le(&buf, 20) // version needed
		write_u16_le(&buf, 0) // flags
		write_u16_le(&buf, 0) // compression = stored
		write_u16_le(&buf, 0) // mod time
		write_u16_le(&buf, 0) // mod date

		crc := hash.crc32(data_bytes)
		write_u32_le(&buf, crc)

		write_u32_le(&buf, u32(len(data_bytes))) // compressed size
		write_u32_le(&buf, u32(len(data_bytes))) // uncompressed size

		write_u16_le(&buf, u16(len(name_bytes))) // filename length
		write_u16_le(&buf, 0) // extra length

		bytes.buffer_write(&buf, name_bytes)
		bytes.buffer_write(&buf, data_bytes)
	}

	// ---- Central directory ----
	central_dir_start := bytes.buffer_length(&buf)

	for i in 0 ..< len(entries) {
		e := entries[i]
		name_bytes := transmute([]u8)e.name
		data_bytes := e.data
		crc := hash.crc32(data_bytes)

		write_u32_le(&buf, 0x02014b50) // central header sig
		write_u16_le(&buf, 20) // version made by
		write_u16_le(&buf, 20) // version needed
		write_u16_le(&buf, 0) // flags
		write_u16_le(&buf, 0) // compression
		write_u16_le(&buf, 0) // mod time
		write_u16_le(&buf, 0) // mod date
		write_u32_le(&buf, crc)
		write_u32_le(&buf, u32(len(data_bytes)))
		write_u32_le(&buf, u32(len(data_bytes)))
		write_u16_le(&buf, u16(len(name_bytes)))
		write_u16_le(&buf, 0) // extra
		write_u16_le(&buf, 0) // comment
		write_u16_le(&buf, 0) // disk start
		write_u16_le(&buf, 0) // internal attrs
		write_u32_le(&buf, 0) // external attrs
		write_u32_le(&buf, u32(local_offsets[i]))

		bytes.buffer_write(&buf, name_bytes)
	}

	central_dir_size := bytes.buffer_length(&buf) - central_dir_start

	// ---- EOCD ----
	write_u32_le(&buf, 0x06054b50)
	write_u16_le(&buf, 0)
	write_u16_le(&buf, 0)
	write_u16_le(&buf, u16(len(entries)))
	write_u16_le(&buf, u16(len(entries)))
	write_u32_le(&buf, u32(central_dir_size))
	write_u32_le(&buf, u32(central_dir_start))
	write_u16_le(&buf, 0) // comment length

	// ---- Write to disk ----
	data := bytes.buffer_to_bytes(&buf)
	err := os.write_entire_file(path, data)
	return err == nil
}
