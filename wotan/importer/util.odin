package importer

import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:unicode/utf8"


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

// HTML 4 Latin-1 entity names for code points 160..255, in order
@(private = "file")
LATIN1_ENTITIES := [96]string {
	"nbsp", "iexcl", "cent", "pound", "curren", "yen", "brvbar", "sect",
	"uml", "copy", "ordf", "laquo", "not", "shy", "reg", "macr",
	"deg", "plusmn", "sup2", "sup3", "acute", "micro", "para", "middot",
	"cedil", "sup1", "ordm", "raquo", "frac14", "frac12", "frac34", "iquest",
	"Agrave", "Aacute", "Acirc", "Atilde", "Auml", "Aring", "AElig", "Ccedil",
	"Egrave", "Eacute", "Ecirc", "Euml", "Igrave", "Iacute", "Icirc", "Iuml",
	"ETH", "Ntilde", "Ograve", "Oacute", "Ocirc", "Otilde", "Ouml", "times",
	"Oslash", "Ugrave", "Uacute", "Ucirc", "Uuml", "Yacute", "THORN", "szlig",
	"agrave", "aacute", "acirc", "atilde", "auml", "aring", "aelig", "ccedil",
	"egrave", "eacute", "ecirc", "euml", "igrave", "iacute", "icirc", "iuml",
	"eth", "ntilde", "ograve", "oacute", "ocirc", "otilde", "ouml", "divide",
	"oslash", "ugrave", "uacute", "ucirc", "uuml", "yacute", "thorn", "yuml",
}

@(private = "file")
_named_entity :: proc(name: string) -> (rune, bool) {
	switch name {
	case "amp":
		return '&', true
	case "lt":
		return '<', true
	case "gt":
		return '>', true
	case "quot":
		return '"', true
	case "apos":
		return '\'', true
	case "nbsp":
		return ' ', true // plain space: friendlier for data than U+00A0
	case "euro":
		return 0x20AC, true
	case "ndash":
		return 0x2013, true
	case "mdash":
		return 0x2014, true
	case "lsquo":
		return 0x2018, true
	case "rsquo":
		return 0x2019, true
	case "ldquo":
		return 0x201C, true
	case "rdquo":
		return 0x201D, true
	case "bull":
		return 0x2022, true
	case "hellip":
		return 0x2026, true
	case "trade":
		return 0x2122, true
	case "minus":
		return 0x2212, true
	}
	for n, i in LATIN1_ENTITIES {
		if n == name {return rune(160 + i), true}
	}
	return 0, false
}

// Decode HTML / XML character references in one pass: named entities
// (&amp; &lt; ... and the HTML Latin-1 set), decimal &#233; and hex &#xE9;.
// Each reference is decoded exactly once ("&amp;lt;" -> "&lt;"); unknown or
// malformed references are kept verbatim. Returns `s` itself when it has no '&'.
decode_entities :: proc(s: string, allocator := context.temp_allocator) -> string {
	if strings.index_byte(s, '&') < 0 {return s}
	b := strings.builder_make(0, len(s), allocator)
	i := 0
	for i < len(s) {
		if s[i] != '&' {
			strings.write_byte(&b, s[i])
			i += 1
			continue
		}
		semi := strings.index_byte(s[i:], ';')
		if semi < 2 || semi > 12 {
			strings.write_byte(&b, '&')
			i += 1
			continue
		}
		name := s[i + 1:i + semi]
		r: rune
		ok := false
		if name[0] == '#' {
			v: u64
			if len(name) > 2 && (name[1] == 'x' || name[1] == 'X') {
				v, ok = strconv.parse_u64_of_base(name[2:], 16)
			} else {
				v, ok = strconv.parse_u64_of_base(name[1:], 10)
			}
			ok = ok && v > 0 && v <= 0x10FFFF && !(v >= 0xD800 && v <= 0xDFFF)
			r = rune(v)
		} else {
			r, ok = _named_entity(name)
		}
		if !ok {
			strings.write_byte(&b, '&')
			i += 1
			continue
		}
		buf, n := utf8.encode_rune(r)
		strings.write_bytes(&b, buf[:n])
		i += semi + 1
	}
	return strings.to_string(b)
}
