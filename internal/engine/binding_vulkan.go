//go:build vulkan

package engine

// The Vulkan build's libraries. In a file of their own, after binding.go, so
// they land after -lggml on the link line: static archives resolve left to
// right, and ggml's backend registry is what references ggml-vulkan.
//
// On Windows the loader is taken by its import library's file name: the
// package links with -static, which would otherwise look for a static
// vulkan-1 that does not exist. vulkan-1.dll itself comes with the driver.

/*
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/ggml/src/ggml-vulkan -lggml-vulkan -lggml-base -lstdc++
#cgo linux LDFLAGS: -lvulkan
#cgo windows LDFLAGS: -l:libvulkan-1.dll.a
*/
import "C"

const buildBackend = "vulkan"
