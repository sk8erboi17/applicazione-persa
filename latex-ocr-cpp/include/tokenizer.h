#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace latex_ocr {

// BPE Tokenizer that loads from HuggingFace tokenizer.json
class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Load from tokenizer.json
    bool load(const std::string& json_path);

    // Encode text to token IDs
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<int>& ids) const;

    // Decode single token ID
    std::string decode_token(int id) const;

    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    int pad_token_id() const { return pad_id_; }
    int bos_token_id() const { return bos_id_; }
    int eos_token_id() const { return eos_id_; }

private:
    // Vocabulary
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;

    // BPE merge rules
    struct MergePair {
        std::string left;
        std::string right;
    };
    std::vector<MergePair> merges_;
    std::unordered_map<std::string, int> merge_ranks_;

    // ByteLevel pre-tokenizer
    std::string byte_to_unicode_[256];
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;

    // Special tokens
    int pad_id_ = 0;
    int bos_id_ = 1;
    int eos_id_ = 2;

    void init_byte_level_mapping();
    std::vector<std::string> bpe_encode(const std::string& word) const;
    std::vector<std::string> pre_tokenize(const std::string& text) const;
};

} // namespace latex_ocr
