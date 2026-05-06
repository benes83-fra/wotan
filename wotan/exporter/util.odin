package exporter

import "core:os"


write_file :: proc(path: string, contents: string) -> bool {
	flags := os.O_CREATE + os.O_TRUNC + os.O_WRONLY
	perm := os.perm_number(0o644)

	f, err := os.open(path, flags, perm)
	if err != nil {
		return false
	}
	defer os.close(f)

	// SAFE here because we only *read* the bytes
	bytes := transmute([]u8)contents

	n, err2 := os.write(f, bytes)
	return err2 == nil && n == len(bytes)
}
