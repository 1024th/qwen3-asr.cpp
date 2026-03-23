#pragma once

#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace qwen3_asr {

inline void append_utf8_codepoint(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

inline void skip_json_ws(const std::string & text, size_t & pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
        ++pos;
    }
}

inline bool parse_json_string(const std::string & text, size_t & pos, std::string & out, std::string & error) {
    if (pos >= text.size() || text[pos] != '"') {
        error = "Expected JSON string";
        return false;
    }

    ++pos;
    out.clear();

    while (pos < text.size()) {
        const char ch = text[pos++];
        if (ch == '"') {
            return true;
        }
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }

        if (pos >= text.size()) {
            error = "Unexpected end of escape sequence";
            return false;
        }

        const char esc = text[pos++];
        switch (esc) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > text.size()) {
                    error = "Truncated \\u escape";
                    return false;
                }
                uint32_t cp = 0;
                for (int i = 0; i < 4; ++i) {
                    const char hex = text[pos++];
                    cp <<= 4;
                    if (hex >= '0' && hex <= '9') cp |= static_cast<uint32_t>(hex - '0');
                    else if (hex >= 'a' && hex <= 'f') cp |= static_cast<uint32_t>(hex - 'a' + 10);
                    else if (hex >= 'A' && hex <= 'F') cp |= static_cast<uint32_t>(hex - 'A' + 10);
                    else {
                        error = "Invalid hex digit in \\u escape";
                        return false;
                    }
                }
                append_utf8_codepoint(out, cp);
                break;
            }
            default:
                error = "Unsupported JSON escape";
                return false;
        }
    }

    error = "Unterminated JSON string";
    return false;
}

inline bool load_vocab_from_json_file(const std::string & path, std::vector<std::string> & vocab, std::string & error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "Failed to open vocab JSON: " + path;
        return false;
    }

    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    skip_json_ws(text, pos);
    if (pos >= text.size() || text[pos] != '{') {
        error = "Expected JSON object in vocab file: " + path;
        return false;
    }
    ++pos;

    std::vector<std::pair<int32_t, std::string>> entries;

    while (true) {
        skip_json_ws(text, pos);
        if (pos >= text.size()) {
            error = "Unexpected end of vocab JSON";
            return false;
        }
        if (text[pos] == '}') {
            ++pos;
            break;
        }

        std::string token;
        if (!parse_json_string(text, pos, token, error)) {
            error = path + ": " + error;
            return false;
        }

        skip_json_ws(text, pos);
        if (pos >= text.size() || text[pos] != ':') {
            error = path + ": expected ':' after token";
            return false;
        }
        ++pos;

        skip_json_ws(text, pos);
        if (pos >= text.size()) {
            error = path + ": missing token id";
            return false;
        }

        bool negative = false;
        if (text[pos] == '-') {
            negative = true;
            ++pos;
        }
        if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
            error = path + ": invalid token id";
            return false;
        }

        int64_t value = 0;
        while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])) != 0) {
            value = value * 10 + (text[pos] - '0');
            if (value > std::numeric_limits<int32_t>::max()) {
                error = path + ": token id out of range";
                return false;
            }
            ++pos;
        }
        if (negative) {
            value = -value;
        }
        if (value < 0) {
            error = path + ": negative token id not supported";
            return false;
        }

        entries.emplace_back(static_cast<int32_t>(value), std::move(token));

        skip_json_ws(text, pos);
        if (pos >= text.size()) {
            error = path + ": unexpected end after token id";
            return false;
        }
        if (text[pos] == ',') {
            ++pos;
            continue;
        }
        if (text[pos] == '}') {
            ++pos;
            break;
        }

        error = path + ": expected ',' or '}'";
        return false;
    }

    int32_t max_id = -1;
    for (const auto & entry : entries) {
        if (entry.first > max_id) {
            max_id = entry.first;
        }
    }
    if (max_id < 0) {
        error = path + ": empty vocab";
        return false;
    }

    vocab.assign(static_cast<size_t>(max_id + 1), "");
    for (const auto & entry : entries) {
        vocab[static_cast<size_t>(entry.first)] = entry.second;
    }

    return true;
}

} // namespace qwen3_asr
