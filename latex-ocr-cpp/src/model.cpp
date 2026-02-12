#include "model.h"
#include "encoder.h"
#include "decoder.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef LATEX_OCR_USE_METAL
#include "ggml-metal.h"
#endif

#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdio>

namespace latex_ocr {

ModelState::~ModelState() {
    if (buffer) ggml_backend_buffer_free(buffer);
    if (ctx_weights) ggml_free(ctx_weights);
    if (backend) ggml_backend_free(backend);
}

// GGUF file parsing
struct GGUFHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
};

static bool read_gguf_string(FILE* f, std::string& out) {
    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) return false;
    out.resize(len);
    if (len > 0 && fread(&out[0], 1, len, f) != len) return false;
    return true;
}

static bool skip_gguf_value(FILE* f, uint32_t type) {
    switch (type) {
        case 0: fseek(f, 1, SEEK_CUR); break;   // UINT8
        case 1: fseek(f, 1, SEEK_CUR); break;   // INT8
        case 2: fseek(f, 2, SEEK_CUR); break;   // UINT16
        case 3: fseek(f, 2, SEEK_CUR); break;   // INT16
        case 4: fseek(f, 4, SEEK_CUR); break;   // UINT32
        case 5: fseek(f, 4, SEEK_CUR); break;   // INT32
        case 6: fseek(f, 4, SEEK_CUR); break;   // FLOAT32
        case 7: fseek(f, 1, SEEK_CUR); break;   // BOOL
        case 8: { std::string s; read_gguf_string(f, s); break; }  // STRING
        case 9: {  // ARRAY
            uint32_t arr_type;
            uint64_t arr_len;
            if (fread(&arr_type, 4, 1, f) != 1) return false;
            if (fread(&arr_len, 8, 1, f) != 1) return false;
            for (uint64_t i = 0; i < arr_len; i++) {
                if (!skip_gguf_value(f, arr_type)) return false;
            }
            break;
        }
        case 10: fseek(f, 8, SEEK_CUR); break;  // UINT64
        case 11: fseek(f, 8, SEEK_CUR); break;  // INT64
        case 12: fseek(f, 8, SEEK_CUR); break;  // FLOAT64
        default: return false;
    }
    return true;
}

static bool read_gguf_uint32(FILE* f, const std::string& key, uint32_t& out, Config& config) {
    // This is called after we've identified the key and type
    (void)key;
    (void)config;
    return fread(&out, sizeof(out), 1, f) == 1;
}

ModelState* load_model(const std::string& gguf_path, bool use_gpu) {
    FILE* f = fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "Cannot open model file: " << gguf_path << std::endl;
        return nullptr;
    }

    // Read header
    GGUFHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        std::cerr << "Failed to read GGUF header" << std::endl;
        fclose(f);
        return nullptr;
    }

    if (header.magic != 0x46475547) {  // "GGUF"
        std::cerr << "Invalid GGUF magic number" << std::endl;
        fclose(f);
        return nullptr;
    }

    std::cout << "GGUF version: " << header.version << std::endl;
    std::cout << "Tensors: " << header.n_tensors << std::endl;
    std::cout << "Metadata entries: " << header.n_kv << std::endl;

    auto* state = new ModelState();

    // Read metadata key-value pairs
    for (uint64_t i = 0; i < header.n_kv; i++) {
        std::string key;
        uint32_t type;
        if (!read_gguf_string(f, key)) break;
        if (fread(&type, 4, 1, f) != 1) break;

        // Parse known keys
        if (key == "latex_ocr.embedding_length" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.dim = val;
        } else if (key == "latex_ocr.encoder.depth" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.encoder_depth = val;
        } else if (key == "latex_ocr.decoder.depth" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.decoder_depth = val;
        } else if (key == "latex_ocr.attention.head_count" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.heads = val;
        } else if (key == "latex_ocr.vocab_size" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.vocab_size = val;
        } else if (key == "latex_ocr.patch_size" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.patch_size = val;
        } else if (key == "latex_ocr.context_length" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.max_seq_len = val;
        } else if (key == "latex_ocr.image.max_width" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.max_width = val;
        } else if (key == "latex_ocr.image.max_height" && type == 4) {
            uint32_t val;
            fread(&val, 4, 1, f);
            state->config.max_height = val;
        } else if (key == "latex_ocr.temperature" && type == 6) {
            float val;
            fread(&val, 4, 1, f);
            state->config.temperature = val;
        } else if (key == "latex_ocr.decoder.attn_on_attn" && type == 7) {
            bool val;
            fread(&val, 1, 1, f);
            state->config.attn_on_attn = val;
        } else if (key == "latex_ocr.decoder.ff_glu" && type == 7) {
            bool val;
            fread(&val, 1, 1, f);
            state->config.ff_glu = val;
        } else if (key == "latex_ocr.decoder.cross_attend" && type == 7) {
            bool val;
            fread(&val, 1, 1, f);
            state->config.cross_attend = val;
        } else {
            skip_gguf_value(f, type);
        }
    }

    // Read tensor info
    struct TensorInfo {
        std::string name;
        uint32_t ndim;
        std::vector<uint64_t> shape;
        uint32_t type;
        uint64_t offset;
    };

    std::vector<TensorInfo> tensor_infos;
    size_t total_size = 0;

    for (uint64_t i = 0; i < header.n_tensors; i++) {
        TensorInfo info;
        if (!read_gguf_string(f, info.name)) break;
        if (fread(&info.ndim, 4, 1, f) != 1) break;

        info.shape.resize(info.ndim);
        for (uint32_t d = 0; d < info.ndim; d++) {
            if (fread(&info.shape[d], 8, 1, f) != 1) break;
        }

        if (fread(&info.type, 4, 1, f) != 1) break;
        if (fread(&info.offset, 8, 1, f) != 1) break;

        // Calculate size
        size_t numel = 1;
        for (uint32_t d = 0; d < info.ndim; d++) numel *= info.shape[d];

        size_t elem_size = 4;  // FP32
        if (info.type == 1) elem_size = 2;  // FP16
        else if (info.type == 8) elem_size = 1;  // Q8_0 approx

        total_size += numel * elem_size;
        state->total_params += numel;
        tensor_infos.push_back(info);
    }

    state->model_size_bytes = total_size;

    // Align to 32 bytes for data section
    long current_pos = ftell(f);
    long aligned_pos = (current_pos + 31) & ~31;
    fseek(f, aligned_pos, SEEK_SET);
    long data_start = ftell(f);

    // Initialize backend
#ifdef LATEX_OCR_USE_METAL
    if (use_gpu) {
        state->backend = ggml_backend_metal_init();
        if (state->backend) {
            std::cout << "Using Metal GPU backend" << std::endl;
        }
    }
#endif
    if (!state->backend) {
        state->backend = ggml_backend_cpu_init();
        std::cout << "Using CPU backend" << std::endl;
        ggml_backend_cpu_set_n_threads(state->backend, state->config.n_threads);
    }

    // Create ggml context for weights
    struct ggml_init_params ctx_params = {
        .mem_size = ggml_tensor_overhead() * header.n_tensors + 1024 * 1024,
        .mem_buffer = nullptr,
        .no_alloc = true,
    };
    state->ctx_weights = ggml_init(ctx_params);

    // Create tensors
    for (const auto& info : tensor_infos) {
        enum ggml_type gtype = GGML_TYPE_F32;
        if (info.type == 1) gtype = GGML_TYPE_F16;
        else if (info.type == 8) gtype = GGML_TYPE_Q8_0;

        struct ggml_tensor* tensor = nullptr;
        switch (info.ndim) {
            case 1:
                tensor = ggml_new_tensor_1d(state->ctx_weights, gtype, info.shape[0]);
                break;
            case 2:
                tensor = ggml_new_tensor_2d(state->ctx_weights, gtype, info.shape[0], info.shape[1]);
                break;
            case 3:
                tensor = ggml_new_tensor_3d(state->ctx_weights, gtype, info.shape[0], info.shape[1], info.shape[2]);
                break;
            case 4:
                tensor = ggml_new_tensor_4d(state->ctx_weights, gtype, info.shape[0], info.shape[1], info.shape[2], info.shape[3]);
                break;
            default:
                std::cerr << "Unsupported tensor ndim: " << info.ndim << std::endl;
                continue;
        }

        if (tensor) {
            ggml_set_name(tensor, info.name.c_str());
            state->weights.tensors[info.name] = tensor;
        }
    }

    // Allocate buffer and load data
    state->buffer = ggml_backend_alloc_ctx_tensors(state->ctx_weights, state->backend);

    // Load tensor data from file
    for (const auto& info : tensor_infos) {
        auto it = state->weights.tensors.find(info.name);
        if (it == state->weights.tensors.end()) continue;

        struct ggml_tensor* tensor = it->second;
        size_t nbytes = ggml_nbytes(tensor);

        // Seek to tensor data in file
        fseek(f, data_start + info.offset, SEEK_SET);

        // Read into temporary buffer then copy to backend
        std::vector<char> buf(nbytes);
        if (fread(buf.data(), 1, nbytes, f) == nbytes) {
            ggml_backend_tensor_set(tensor, buf.data(), 0, nbytes);
        }
    }

    fclose(f);

    std::cout << "Model loaded: " << state->total_params << " parameters ("
              << state->model_size_bytes / (1024 * 1024) << " MB)" << std::endl;

    return state;
}

void free_model(ModelState* state) {
    delete state;
}

std::vector<int> run_inference(
    ModelState* model,
    const float* image_pixels,
    int image_width,
    int image_height,
    float temperature,
    int max_tokens
) {
    if (!model) return {};

    const Config& config = model->config;

    // Create computation context
    size_t ctx_size = 512 * 1024 * 1024;  // 512MB for computation
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = nullptr,
        .no_alloc = false,
    };
    struct ggml_context* ctx = ggml_init(params);

    // Create input image tensor [1, 1, H, W]
    struct ggml_tensor* images = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                     image_width, image_height, 1, 1);
    memcpy(images->data, image_pixels, image_width * image_height * sizeof(float));

    // Build encoder graph
    struct ggml_tensor* encoded = build_encoder_graph(
        ctx, images, model->weights, config, image_height, image_width
    );

    // Build computation graph and evaluate
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, encoded);
    ggml_backend_graph_compute(model->backend, gf);

    // Autoregressive decoding
    std::vector<int> tokens;
    tokens.push_back(config.bos_token);

    KVCache kv_cache;
    kv_cache.k = nullptr;
    kv_cache.v = nullptr;
    kv_cache.current_len = 0;
    kv_cache.max_len = config.max_seq_len;

    for (int pos = 0; pos < max_tokens; pos++) {
        int current_token = tokens.back();

        // Look up token embedding
        struct ggml_tensor* tok_emb_table = model->weights.get("decoder.net.token_emb.emb.weight");
        if (!tok_emb_table) break;

        struct ggml_tensor* tok_emb = ggml_view_2d(ctx, tok_emb_table,
                                                     config.dim, 1,
                                                     tok_emb_table->nb[1],
                                                     current_token * config.dim * ggml_type_size(tok_emb_table->type));

        // Build decoder step
        struct ggml_tensor* logits = build_decoder_step(
            ctx, tok_emb, encoded, model->weights, kv_cache, config, pos
        );

        // Evaluate
        struct ggml_cgraph* dec_graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(dec_graph, logits);
        ggml_backend_graph_compute(model->backend, dec_graph);

        // Get logits data
        std::vector<float> logits_data(config.vocab_size);
        ggml_backend_tensor_get(logits, logits_data.data(), 0,
                                 config.vocab_size * sizeof(float));

        // Sample next token
        int next_token = sample_top_k(logits_data.data(), config.vocab_size,
                                       temperature, config.top_k);

        if (next_token == config.eos_token) break;
        tokens.push_back(next_token);
    }

    ggml_free(ctx);

    // Remove BOS token from output
    if (!tokens.empty() && tokens[0] == config.bos_token) {
        tokens.erase(tokens.begin());
    }

    return tokens;
}

void print_model_info(const ModelState* model) {
    if (!model) return;

    const Config& config = model->config;
    std::cout << "=== LaTeX-OCR Model ===" << std::endl;
    std::cout << "  Dim: " << config.dim << std::endl;
    std::cout << "  Encoder depth: " << config.encoder_depth << std::endl;
    std::cout << "  Decoder depth: " << config.decoder_depth << std::endl;
    std::cout << "  Heads: " << config.heads << std::endl;
    std::cout << "  Vocab size: " << config.vocab_size << std::endl;
    std::cout << "  Max seq len: " << config.max_seq_len << std::endl;
    std::cout << "  Patch size: " << config.patch_size << std::endl;
    std::cout << "  Image max: " << config.max_width << "x" << config.max_height << std::endl;
    std::cout << "  Parameters: " << model->total_params << std::endl;
    std::cout << "  Size: " << model->model_size_bytes / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Backend: " << (config.use_gpu ? "GPU (Metal)" : "CPU") << std::endl;
}

} // namespace latex_ocr
