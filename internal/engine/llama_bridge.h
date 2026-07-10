#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <stdbool.h>
#include <stdlib.h>

// batch_add is a C helper because llama_batch_add only exists in C++ common/.
static inline void bridge_batch_add(
    struct llama_batch *batch,
    llama_token id,
    llama_pos pos,
    llama_seq_id seq_id,
    bool logits
) {
    int32_t i = batch->n_tokens;
    batch->token[i]      = id;
    batch->pos[i]        = pos;
    batch->n_seq_id[i]   = 1;
    batch->seq_id[i][0]  = seq_id;
    batch->logits[i]     = logits ? 1 : 0;
    batch->n_tokens++;
}

static inline void bridge_batch_clear(struct llama_batch *batch) {
    batch->n_tokens = 0;
}

// chat_apply_template wraps llama_chat_apply_template with a simpler interface.
// Returns the formatted string (caller must free) and its length via out_len.
static inline char *bridge_chat_apply_template(
    const char *tmpl,
    const struct llama_chat_message *msgs,
    size_t n_msgs,
    bool add_ass,
    int32_t *out_len
) {
    // First call to get required size
    int32_t needed = llama_chat_apply_template(tmpl, msgs, n_msgs, add_ass, NULL, 0);
    if (needed < 0) {
        *out_len = needed;
        return NULL;
    }

    char *buf = (char *)malloc(needed + 1);
    if (!buf) {
        *out_len = -1;
        return NULL;
    }

    int32_t written = llama_chat_apply_template(tmpl, msgs, n_msgs, add_ass, buf, needed + 1);
    if (written < 0) {
        free(buf);
        *out_len = written;
        return NULL;
    }

    buf[written] = '\0';
    *out_len = written;
    return buf;
}

typedef struct bridge_schema_grammar_result {
    char *grammar;
    char *error;
} bridge_schema_grammar_result;

typedef struct bridge_mtmd_eval_result {
    int32_t code;
    int32_t n_past;
    char *error;
} bridge_mtmd_eval_result;

typedef struct bridge_chat_render_result {
    char *prompt;
    char *grammar;
    char *parser;
    char *generation_prompt;
    char *additional_stops_json;
    char *grammar_triggers_json;
    char *capabilities_json;
    char *error;
    int32_t format;
    bool grammar_lazy;
    bool supports_thinking;
} bridge_chat_render_result;

typedef struct bridge_chat_parse_result {
    char *message_json;
    char *error;
} bridge_chat_parse_result;

#ifdef __cplusplus
extern "C" {
#endif

bridge_schema_grammar_result bridge_json_schema_to_grammar(const char *schema_json);
void bridge_schema_grammar_result_free(bridge_schema_grammar_result result);
bridge_mtmd_eval_result bridge_mtmd_eval_prompt(
    mtmd_context *mtmd,
    struct llama_context *lctx,
    const char *prompt,
    const unsigned char **image_data,
    const size_t *image_lens,
    size_t n_images,
    bool add_special,
    bool parse_special,
    int32_t n_batch
);
void bridge_mtmd_eval_result_free(bridge_mtmd_eval_result result);
bridge_chat_render_result bridge_chat_render_native(
    const struct llama_model *model,
    const char *template_override,
    const char *messages_json,
    const char *tools_json,
    int32_t tool_choice,
    bool parallel_tool_calls,
    bool enable_thinking
);
void bridge_chat_render_result_free(bridge_chat_render_result result);
bridge_chat_parse_result bridge_chat_parse_native(
    const char *response,
    const char *parser,
    const char *generation_prompt,
    int32_t format
);
void bridge_chat_parse_result_free(bridge_chat_parse_result result);

// Creates the grammar sampler produced by llama.cpp's native chat-template
// pipeline, including lazy tool-call triggers and generation-prompt prefill.
struct llama_sampler * bridge_chat_grammar_sampler_init(
    const struct llama_vocab * vocab,
    const char * grammar,
    bool grammar_lazy,
    const char * grammar_triggers_json,
    const char * generation_prompt,
    char ** error);
void bridge_string_free(char * value);

struct common_sampler;
struct common_sampler * bridge_common_sampler_init(
    const struct llama_model * model,
    const char * options_json,
    const char * grammar,
    bool grammar_lazy,
    const char * grammar_triggers_json,
    const char * generation_prompt,
    char ** error);
llama_token bridge_common_sampler_sample(struct common_sampler * sampler, struct llama_context * ctx, int32_t idx);
void bridge_common_sampler_accept(struct common_sampler * sampler, llama_token token);
void bridge_common_sampler_free(struct common_sampler * sampler);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_BRIDGE_H
