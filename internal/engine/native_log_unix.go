//go:build !windows

package engine

import (
	"os"
	"syscall"
)

// nativeLogWriteFD returns the descriptor the C log callback writes to.
func nativeLogWriteFD(w *os.File) (int, error) {
	// Fd() switches the descriptor to blocking mode, so set non-blocking after.
	fd := int(w.Fd())
	if err := syscall.SetNonblock(fd, true); err != nil {
		return -1, err
	}
	return fd, nil
}
