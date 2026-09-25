package engine

/*
#include "ggml-backend.h"
*/
import "C"

// GPUDevice is a graphics device the linked llama.cpp backends can compute on.
type GPUDevice struct {
	Name        string
	Description string
	Backend     string // the backend's own name: Vulkan, CUDA, Metal
	TotalBytes  uint64
	FreeBytes   uint64
}

// BuildBackend is the graphics backend this binary was built with: "vulkan",
// or "" for the processor build (and for macOS, whose Metal is reported as
// before).
func BuildBackend() string { return buildBackend }

// GPUDevices lists the GPUs llama.cpp itself sees — what a model will be
// offloaded to — rather than guessing from nvidia-smi. Empty for a processor
// build, or when the driver offers no usable device.
func GPUDevices() []GPUDevice {
	var out []GPUDevice
	count := int(C.ggml_backend_dev_count())
	for i := 0; i < count; i++ {
		dev := C.ggml_backend_dev_get(C.size_t(i))
		kind := C.ggml_backend_dev_type(dev)
		if kind != C.GGML_BACKEND_DEVICE_TYPE_GPU && kind != C.GGML_BACKEND_DEVICE_TYPE_IGPU {
			continue
		}
		var free, total C.size_t
		C.ggml_backend_dev_memory(dev, &free, &total)
		out = append(out, GPUDevice{
			Name:        C.GoString(C.ggml_backend_dev_name(dev)),
			Description: C.GoString(C.ggml_backend_dev_description(dev)),
			Backend:     C.GoString(C.ggml_backend_reg_name(C.ggml_backend_dev_backend_reg(dev))),
			TotalBytes:  uint64(total),
			FreeBytes:   uint64(free),
		})
	}
	return out
}
