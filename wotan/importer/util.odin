package importer

import "core:os"
import os2 "core:os/os2"
import "core:strings"




read_file :: proc (file: string) -> (string,os2.Error) {
    

    when ODIN_VERSION == "dev-2026-02" {
        data, err := os.read_entire_file(file,context.allocator)
        defer delete(data,context.allocator)
       
        return strings.clone_from_bytes(data,context.allocator)
    } else {
        data, err := os2.read_entire_file(file,context.allocator)
        defer delete(data,context.allocatior)
        if err!= nil {
            return "", err
        
        return strings.clone_from_bytes(data,context.allocator)
        }
    }

}