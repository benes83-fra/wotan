package importer

import "core:os"


import os2 "core:os/os2"


import "core:strings"


// this module exists for compatibility reason with future API changes in the os module. Once the os2 way becomes the new default way, replace os2 by os

read_file :: proc(file: string) -> (string, os2.Error) {


	data, err := os2.read_entire_file(file, context.allocator)
	defer delete(data, context.allocator)
	if err != nil {
		return "", err
	}
	ret := strings.clone_from_bytes(data, context.allocator)
	return ret, err


}
