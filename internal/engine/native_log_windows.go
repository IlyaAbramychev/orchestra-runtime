//go:build windows

package engine

/*
#include <windows.h>
#include <io.h>
#include <fcntl.h>

// The C log callback uses write(), which takes a CRT descriptor, not a
// HANDLE. PIPE_NOWAIT keeps a slow reader from stalling inference threads,
// matching O_NONBLOCK on Unix.
static int bridge_pipe_handle_to_fd(void *h) {
	DWORD mode = PIPE_READMODE_BYTE | PIPE_NOWAIT;
	if (!SetNamedPipeHandleState((HANDLE) h, &mode, NULL, NULL)) {
		return -1;
	}
	return _open_osfhandle((intptr_t) h, _O_WRONLY | _O_BINARY);
}
*/
import "C"

import (
	"errors"
	"os"
	"unsafe"
)

// nativeLogWriteFD returns the descriptor the C log callback writes to.
func nativeLogWriteFD(w *os.File) (int, error) {
	fd := int(C.bridge_pipe_handle_to_fd(unsafe.Pointer(w.Fd())))
	if fd < 0 {
		return -1, errors.New("cannot open native log pipe")
	}
	return fd, nil
}
