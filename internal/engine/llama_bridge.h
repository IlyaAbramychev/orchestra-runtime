#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#include "llama.h"
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

#endif // LLAMA_BRIDGE_H
