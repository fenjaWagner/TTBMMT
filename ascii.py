def convert_to_ascii(format_string):

    ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ascii_index = 0
    char_mapping = {}
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
                ascii_index += 1
            ascii_expression_parts.append(char_mapping[char])
    print(char_mapping)
    return "".join(ascii_expression_parts)
