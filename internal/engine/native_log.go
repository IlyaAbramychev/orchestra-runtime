package engine

/*
void bridge_install_native_logger(int fd);
*/
import "C"

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"os"
	"syscall"
)

// ggml_log_level values from ggml.h.
const (
	ggmlLogDebug = 1
	ggmlLogInfo  = 2
	ggmlLogWarn  = 3
	ggmlLogError = 4
	ggmlLogCont  = 5

	nativeLogRecordSep = 0x1e
	// Flush a line that never receives a newline instead of growing forever.
	nativeLogMaxLine = 64 * 1024
)

// nativeLogWriter keeps the pipe's write end referenced so its finalizer never
// closes the descriptor the C callback writes to.
var nativeLogWriter *os.File

// installNativeLogger replaces llama.cpp's default stderr logging with
// structured slog records. Without it, thousands of raw lines are interleaved
// with the JSON log stream and never reach /api/logs.
func installNativeLogger() {
	r, w, err := os.Pipe()
	if err != nil {
		slog.Warn("native log capture disabled", "error", err)
		return
	}
	// Fd() switches the descriptor to blocking mode, so set non-blocking after.
	fd := int(w.Fd())
	if err := syscall.SetNonblock(fd, true); err != nil {
		slog.Warn("native log capture disabled", "error", err)
		_ = r.Close()
		_ = w.Close()
		return
	}
	nativeLogWriter = w
	go pumpNativeLog(r, emitNativeLog)
	C.bridge_install_native_logger(C.int(fd))
}

func emitNativeLog(level slog.Level, msg string) {
	slog.Log(context.Background(), level, msg, "component", "llama.cpp")
}

// pumpNativeLog decodes records written by bridge_native_log: a separator
// byte, a level digit, then text that may hold several lines or a fragment
// continued by a later GGML_LOG_LEVEL_CONT record.
func pumpNativeLog(r io.Reader, emit func(slog.Level, string)) {
	asm := nativeLogAssembler{emit: emit, level: slog.LevelDebug}
	buf := make([]byte, 16*1024)
	expectLevel := false
	for {
		n, err := r.Read(buf)
		chunk := buf[:n]
		for len(chunk) > 0 {
			if expectLevel {
				asm.begin(int(chunk[0] - '0'))
				expectLevel = false
				chunk = chunk[1:]
				continue
			}
			i := bytes.IndexByte(chunk, nativeLogRecordSep)
			if i < 0 {
				asm.text(chunk)
				break
			}
			asm.text(chunk[:i])
			expectLevel = true
			chunk = chunk[i+1:]
		}
		if err != nil {
			asm.flush()
			return
		}
	}
}

type nativeLogAssembler struct {
	line  []byte
	level slog.Level
	emit  func(slog.Level, string)
}

func (a *nativeLogAssembler) begin(level int) {
	if level == ggmlLogCont {
		return
	}
	// A new record terminates any unfinished line of the previous one.
	a.flush()
	a.level = nativeSlogLevel(level)
}

func (a *nativeLogAssembler) text(p []byte) {
	for len(p) > 0 {
		i := bytes.IndexByte(p, '\n')
		if i < 0 {
			a.line = append(a.line, p...)
			if len(a.line) >= nativeLogMaxLine {
				a.flush()
			}
			return
		}
		a.line = append(a.line, p[:i]...)
		a.flush()
		p = p[i+1:]
	}
}

func (a *nativeLogAssembler) flush() {
	msg := bytes.TrimSpace(a.line)
	a.line = a.line[:0]
	if len(msg) > 0 {
		a.emit(a.level, string(msg))
	}
}

// nativeSlogLevel demotes llama.cpp INFO to debug: it reports every tensor,
// Metal kernel and KV buffer. Warnings and errors keep their level so load
// failures stay visible at the default log level.
func nativeSlogLevel(level int) slog.Level {
	switch level {
	case ggmlLogWarn:
		return slog.LevelWarn
	case ggmlLogError:
		return slog.LevelError
	default:
		return slog.LevelDebug
	}
}
