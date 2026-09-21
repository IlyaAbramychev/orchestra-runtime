// Routes llama.cpp, ggml and mtmd logging into a pipe that Go drains into
// slog (see native_log.go).
//
// The callback deliberately does not call into Go: ggml logs from Metal
// worker threads and from destructors that run during process exit, where a
// cgo callback can deadlock or abort. A non-blocking write to a pipe is safe
// in all of those contexts; if the reader falls behind, logs are dropped
// instead of stalling inference threads.

#include <pthread.h>
#include <string.h>
#include <unistd.h>

#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

// Record separator; the byte after it is the ggml log level as '0'..'5'.
#define BRIDGE_LOG_RECORD_SEP '\x1e'

static int g_log_fd = -1;
static pthread_mutex_t g_log_mu = PTHREAD_MUTEX_INITIALIZER;

static void bridge_native_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    int fd = g_log_fd;
    if (fd < 0 || text == NULL) {
        return;
    }
    char head[2] = { BRIDGE_LOG_RECORD_SEP, (char) ('0' + (int) level) };
    size_t len = strlen(text);

    pthread_mutex_lock(&g_log_mu);
    ssize_t n = write(fd, head, sizeof head);
    if (n == (ssize_t) sizeof head && len > 0) {
        n = write(fd, text, len);
    }
    (void) n;
    pthread_mutex_unlock(&g_log_mu);
}

void bridge_install_native_logger(int fd) {
    g_log_fd = fd;
    llama_log_set(bridge_native_log, NULL);
    mtmd_log_set(bridge_native_log, NULL);
}
