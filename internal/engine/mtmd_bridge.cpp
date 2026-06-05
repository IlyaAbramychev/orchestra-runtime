#include "llama_bridge.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static char * bridge_mtmd_strdup(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

extern "C" bridge_mtmd_eval_result bridge_mtmd_eval_prompt(
    mtmd_context * mtmd,
    struct llama_context * lctx,
    const char * prompt,
    const unsigned char ** image_data,
    const size_t * image_lens,
    size_t n_images,
    bool add_special,
    bool parse_special,
    int32_t n_batch
) {
    bridge_mtmd_eval_result result = {};
    if (!mtmd) {
        result.code = -1;
        result.error = bridge_mtmd_strdup("mmproj context is not loaded");
        return result;
    }
    if (!lctx) {
        result.code = -1;
        result.error = bridge_mtmd_strdup("llama context is not loaded");
        return result;
    }
    if (!prompt) {
        result.code = -1;
        result.error = bridge_mtmd_strdup("prompt is required");
        return result;
    }
    try {
        std::vector<mtmd_bitmap *> bitmaps;
        bitmaps.reserve(n_images);
        for (size_t i = 0; i < n_images; i++) {
            mtmd_bitmap * bitmap = mtmd_helper_bitmap_init_from_buf(mtmd, image_data[i], image_lens[i]);
            if (!bitmap) {
                for (mtmd_bitmap * prev : bitmaps) {
                    mtmd_bitmap_free(prev);
                }
                result.code = 2;
                result.error = bridge_mtmd_strdup("failed to decode image input");
                return result;
            }
            bitmaps.push_back(bitmap);
        }
        std::vector<const mtmd_bitmap *> bitmap_ptrs;
        bitmap_ptrs.reserve(bitmaps.size());
        for (mtmd_bitmap * bitmap : bitmaps) {
            bitmap_ptrs.push_back(bitmap);
        }

        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (!chunks) {
            for (mtmd_bitmap * bitmap : bitmaps) {
                mtmd_bitmap_free(bitmap);
            }
            result.code = -1;
            result.error = bridge_mtmd_strdup("failed to allocate multimodal input chunks");
            return result;
        }

        mtmd_input_text text {
            /* .text          = */ prompt,
            /* .add_special   = */ add_special,
            /* .parse_special = */ parse_special,
        };

        int32_t tokenize_ret = mtmd_tokenize(
            mtmd,
            chunks,
            &text,
            bitmap_ptrs.empty() ? nullptr : bitmap_ptrs.data(),
            bitmap_ptrs.size()
        );
        for (mtmd_bitmap * bitmap : bitmaps) {
            mtmd_bitmap_free(bitmap);
        }
        if (tokenize_ret != 0) {
            mtmd_input_chunks_free(chunks);
            result.code = tokenize_ret;
            result.error = bridge_mtmd_strdup("failed to tokenize multimodal prompt");
            return result;
        }

        llama_pos n_past = 0;
        int32_t eval_ret = mtmd_helper_eval_chunks(
            mtmd,
            lctx,
            chunks,
            0,
            0,
            n_batch,
            true,
            &n_past
        );
        mtmd_input_chunks_free(chunks);
        result.code = eval_ret;
        result.n_past = n_past;
        if (eval_ret != 0) {
            result.error = bridge_mtmd_strdup("failed to evaluate multimodal prompt");
        }
        return result;
    } catch (const std::exception & e) {
        result.code = -1;
        result.error = bridge_mtmd_strdup(e.what());
        return result;
    } catch (...) {
        result.code = -1;
        result.error = bridge_mtmd_strdup("unknown multimodal prompt evaluation error");
        return result;
    }
}

extern "C" void bridge_mtmd_eval_result_free(bridge_mtmd_eval_result result) {
    std::free(result.error);
}
