#include "tokenizer.h"
#include "cJSON.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>

namespace latex_ocr {

Tokenizer::Tokenizer() {
    init_byte_level_mapping();
}

Tokenizer::~Tokenizer() = default;

void Tokenizer::init_byte_level_mapping() {
    // ByteLevel pre-tokenizer mapping (same as HuggingFace tokenizers)
    // Maps byte values to unicode characters
    int n = 0;
    // Printable ASCII range that maps to itself
    for (int i = '!'; i <= '~'; i++) {
        byte_to_unicode_[i] = std::string(1, static_cast<char>(i));
        n++;
    }
    for (int i = 0xA1; i <= 0xAC; i++) {
        // Latin-1 Supplement range 1
        byte_to_unicode_[i] = std::string(1, static_cast<char>(i));
        n++;
    }
    for (int i = 0xAE; i <= 0xFF; i++) {
        // Latin-1 Supplement range 2
        byte_to_unicode_[i] = std::string(1, static_cast<char>(i));
        n++;
    }

    // Map remaining bytes to unicode codepoints starting at 256
    int codepoint = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode_[i].empty()) {
            // Encode as UTF-8
            char buf[4];
            int len = 0;
            if (codepoint < 0x80) {
                buf[0] = static_cast<char>(codepoint);
                len = 1;
            } else if (codepoint < 0x800) {
                buf[0] = static_cast<char>(0xC0 | (codepoint >> 6));
                buf[1] = static_cast<char>(0x80 | (codepoint & 0x3F));
                len = 2;
            } else {
                buf[0] = static_cast<char>(0xE0 | (codepoint >> 12));
                buf[1] = static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                buf[2] = static_cast<char>(0x80 | (codepoint & 0x3F));
                len = 3;
            }
            byte_to_unicode_[i] = std::string(buf, len);
            codepoint++;
        }
    }

    // Build reverse mapping
    for (int i = 0; i < 256; i++) {
        unicode_to_byte_[byte_to_unicode_[i]] = static_cast<uint8_t>(i);
    }
}

bool Tokenizer::load(const std::string& json_path) {
    // Read file
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "Cannot open tokenizer file: " << json_path << std::endl;
        return false;
    }

    std::stringstream ss;
    ss << file.rdbuf();
    std::string json_str = ss.str();

    cJSON* root = cJSON_Parse(json_str.c_str());
    if (!root) {
        std::cerr << "Failed to parse tokenizer JSON" << std::endl;
        return false;
    }

    // Parse vocabulary from model.vocab
    cJSON* model = cJSON_GetObjectItem(root, "model");
    if (!model) {
        std::cerr << "Missing 'model' in tokenizer JSON" << std::endl;
        cJSON_Delete(root);
        return false;
    }

    cJSON* vocab = cJSON_GetObjectItem(model, "vocab");
    if (vocab) {
        int max_id = 0;
        cJSON* item;
        cJSON_ArrayForEach(item, vocab) {
            int id = item->valueint;
            if (id > max_id) max_id = id;
        }

        id_to_token_.resize(max_id + 1);
        cJSON_ArrayForEach(item, vocab) {
            std::string token = item->string;
            int id = item->valueint;
            token_to_id_[token] = id;
            id_to_token_[id] = token;
        }
    }

    // Parse merges
    cJSON* merges = cJSON_GetObjectItem(model, "merges");
    if (merges && cJSON_IsArray(merges)) {
        int rank = 0;
        cJSON* merge;
        cJSON_ArrayForEach(merge, merges) {
            if (cJSON_IsString(merge)) {
                std::string merge_str = merge->valuestring;
                size_t space_pos = merge_str.find(' ');
                if (space_pos != std::string::npos) {
                    MergePair pair;
                    pair.left = merge_str.substr(0, space_pos);
                    pair.right = merge_str.substr(space_pos + 1);
                    merges_.push_back(pair);
                    merge_ranks_[merge_str] = rank++;
                }
            }
        }
    }

    // Parse added_tokens for special token IDs
    cJSON* added_tokens = cJSON_GetObjectItem(root, "added_tokens");
    if (added_tokens && cJSON_IsArray(added_tokens)) {
        cJSON* tok;
        cJSON_ArrayForEach(tok, added_tokens) {
            cJSON* content = cJSON_GetObjectItem(tok, "content");
            cJSON* id = cJSON_GetObjectItem(tok, "id");
            if (content && id && cJSON_IsString(content)) {
                std::string token_str = content->valuestring;
                int token_id = id->valueint;
                if (token_str == "[PAD]") pad_id_ = token_id;
                else if (token_str == "[BOS]") bos_id_ = token_id;
                else if (token_str == "[EOS]") eos_id_ = token_id;

                // Add to vocab if not present
                if (token_to_id_.find(token_str) == token_to_id_.end()) {
                    token_to_id_[token_str] = token_id;
                    if (token_id >= static_cast<int>(id_to_token_.size())) {
                        id_to_token_.resize(token_id + 1);
                    }
                    id_to_token_[token_id] = token_str;
                }
            }
        }
    }

    cJSON_Delete(root);

    std::cout << "Tokenizer loaded: " << id_to_token_.size() << " tokens, "
              << merges_.size() << " merges" << std::endl;
    return true;
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    // ByteLevel pre-tokenization: convert each byte to its unicode representation
    std::vector<std::string> result;
    std::string current_word;

    for (size_t i = 0; i < text.size(); i++) {
        uint8_t byte = static_cast<uint8_t>(text[i]);
        current_word += byte_to_unicode_[byte];
    }

    if (!current_word.empty()) {
        result.push_back(current_word);
    }

    return result;
}

std::vector<std::string> Tokenizer::bpe_encode(const std::string& word) const {
    if (word.empty()) return {};

    // Split word into individual UTF-8 characters
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < word.size()) {
        size_t char_len = 1;
        uint8_t c = static_cast<uint8_t>(word[i]);
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        tokens.push_back(word.substr(i, char_len));
        i += char_len;
    }

    if (tokens.size() <= 1) return tokens;

    // Iteratively merge pairs
    while (true) {
        // Find the best merge pair (lowest rank)
        int best_rank = std::numeric_limits<int>::max();
        int best_pos = -1;

        for (size_t j = 0; j + 1 < tokens.size(); j++) {
            std::string pair_key = tokens[j] + " " + tokens[j + 1];
            auto it = merge_ranks_.find(pair_key);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = static_cast<int>(j);
            }
        }

        if (best_pos < 0) break;  // No more merges possible

        // Apply the merge
        std::vector<std::string> new_tokens;
        for (size_t j = 0; j < tokens.size(); j++) {
            if (static_cast<int>(j) == best_pos) {
                new_tokens.push_back(tokens[j] + tokens[j + 1]);
                j++;  // Skip next token (merged)
            } else {
                new_tokens.push_back(tokens[j]);
            }
        }
        tokens = std::move(new_tokens);

        if (tokens.size() <= 1) break;
    }

    return tokens;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;

    // Pre-tokenize
    std::vector<std::string> words = pre_tokenize(text);

    // BPE encode each word
    for (const auto& word : words) {
        std::vector<std::string> bpe_tokens = bpe_encode(word);

        for (const auto& token : bpe_tokens) {
            auto it = token_to_id_.find(token);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
            // Unknown tokens are silently skipped
        }
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string result;

    for (int id : ids) {
        if (id == pad_id_ || id == bos_id_ || id == eos_id_) continue;
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
            result += id_to_token_[id];
        }
    }

    // Reverse ByteLevel encoding: convert unicode chars back to bytes
    std::string decoded;
    size_t i = 0;
    while (i < result.size()) {
        // Get the next UTF-8 character
        size_t char_len = 1;
        uint8_t c = static_cast<uint8_t>(result[i]);
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        std::string unicode_char = result.substr(i, char_len);
        auto it = unicode_to_byte_.find(unicode_char);
        if (it != unicode_to_byte_.end()) {
            decoded += static_cast<char>(it->second);
        } else {
            decoded += unicode_char;  // Pass through unknown
        }
        i += char_len;
    }

    // Clean up: replace Ġ with space
    std::string cleaned;
    for (size_t j = 0; j < decoded.size(); j++) {
        // Ġ is UTF-8: 0xC4 0xA0
        if (j + 1 < decoded.size() &&
            static_cast<uint8_t>(decoded[j]) == 0xC4 &&
            static_cast<uint8_t>(decoded[j+1]) == 0xA0) {
            cleaned += ' ';
            j++;
        } else {
            cleaned += decoded[j];
        }
    }

    return cleaned;
}

std::string Tokenizer::decode_token(int id) const {
    if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
        return id_to_token_[id];
    }
    return "";
}

} // namespace latex_ocr
