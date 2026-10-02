package importer

import "core:mem"
import "core:os"


//import os2 "core:os/os2"


import "core:strings"


// this module exists for compatibility reason with future API changes in the os module. Once the os2 way becomes the new default way, replace os2 by os

read_file :: proc(
	file: string,
	allocator: mem.Allocator = context.allocator,
) -> (
	string,
	os.Error,
) {


	data, err := os.read_entire_file(file, allocator)
	defer delete(data, allocator)
	if err != nil {
		return "", err
	}
	ret := strings.clone_from_bytes(data, allocator)
	return ret, err


}
