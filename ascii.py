def convert_to_ascii(format_string_list):

    ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ascii_index = 0
    char_mapping = {}
    char_mapping_back = {}
    
    for i, format_string in enumerate(format_string_list):
        ascii_expression_parts = []
        for char in format_string:
            if char in ",->":
                ascii_expression_parts.append(char)
            else:
                if char not in char_mapping:
                    if ascii_index == len(ascii_chars):
                        raise RuntimeError(
                            f"ERROR: {format_string} cannot be converted to ASCII, it is too large."
                        )
                    char_mapping[char] = ascii_chars[ascii_index]
                    char_mapping_back[ascii_chars[ascii_index]] = char
                    ascii_index += 1
                ascii_expression_parts.append(char_mapping[char])
        format_string_list[i] = "".join(ascii_expression_parts)
    return format_string_list, char_mapping_back

def convert_ascii_back(format_string_list, char_mapping):
    for i, format_string in enumerate(format_string_list):
        new_format_str = ""
        for char in format_string:
            new_format_str += char_mapping[char]
            """if char in char_mapping:
                new_format_str += char_mapping[char]
            else:
                new_format_str += char"""
        format_string_list[i] = new_format_str

    return format_string_list



