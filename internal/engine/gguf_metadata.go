package engine

/*
#include "gguf.h"
#include <stdlib.h>
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"
)

// GGUFMetadata is the small, load-planning subset of a GGUF header. Reading
// it uses gguf_init_from_file(no_alloc=true), so tensor data is never mapped or
// copied into RAM.
type GGUFMetadata struct {
	Architecture    string
	Name            string
	ChatTemplate    string
	ContextLength   int
	EmbeddingLength int
	ParameterCount  uint64
	PoolingType     int64
	HasPoolingType  bool
}

func ReadGGUFMetadata(path string) (GGUFMetadata, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	params := C.struct_gguf_init_params{no_alloc: C.bool(true), ctx: nil}
	ctx := C.gguf_init_from_file(cPath, params)
	if ctx == nil {
		return GGUFMetadata{}, fmt.Errorf("read GGUF metadata from %s", path)
	}
	defer C.gguf_free(ctx)

	metadata := GGUFMetadata{
		Architecture: ggufString(ctx, "general.architecture"),
		Name:         ggufString(ctx, "general.name"),
		ChatTemplate: ggufString(ctx, "tokenizer.chat_template"),
	}
	if metadata.ChatTemplate == "" {
		metadata.ChatTemplate = ggufFirstString(ctx, "tokenizer.chat_templates")
	}
	if value, ok := ggufUnsigned(ctx, "general.parameter_count"); ok {
		metadata.ParameterCount = value
	}
	if metadata.Architecture != "" {
		if value, ok := ggufUnsigned(ctx, metadata.Architecture+".context_length"); ok && value <= math.MaxInt {
			metadata.ContextLength = int(value)
		}
		if value, ok := ggufUnsigned(ctx, metadata.Architecture+".embedding_length"); ok && value <= math.MaxInt {
			metadata.EmbeddingLength = int(value)
		}
		if value, ok := ggufSigned(ctx, metadata.Architecture+".pooling_type"); ok {
			metadata.PoolingType = value
			metadata.HasPoolingType = true
		}
	}
	return metadata, nil
}

func ggufString(ctx *C.struct_gguf_context, key string) string {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	id := C.gguf_find_key(ctx, cKey)
	if id < 0 || C.gguf_get_kv_type(ctx, id) != C.GGUF_TYPE_STRING {
		return ""
	}
	value := C.gguf_get_val_str(ctx, id)
	if value == nil {
		return ""
	}
	return C.GoString(value)
}

func ggufFirstString(ctx *C.struct_gguf_context, key string) string {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	id := C.gguf_find_key(ctx, cKey)
	if id < 0 || C.gguf_get_kv_type(ctx, id) != C.GGUF_TYPE_ARRAY || C.gguf_get_arr_type(ctx, id) != C.GGUF_TYPE_STRING || C.gguf_get_arr_n(ctx, id) == 0 {
		return ""
	}
	value := C.gguf_get_arr_str(ctx, id, 0)
	if value == nil {
		return ""
	}
	return C.GoString(value)
}

func ggufUnsigned(ctx *C.struct_gguf_context, key string) (uint64, bool) {
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	id := C.gguf_find_key(ctx, cKey)
	if id < 0 {
		return 0, false
	}
	switch C.gguf_get_kv_type(ctx, id) {
	case C.GGUF_TYPE_UINT8:
		return uint64(C.gguf_get_val_u8(ctx, id)), true
	case C.GGUF_TYPE_UINT16:
		return uint64(C.gguf_get_val_u16(ctx, id)), true
	case C.GGUF_TYPE_UINT32:
		return uint64(C.gguf_get_val_u32(ctx, id)), true
	case C.GGUF_TYPE_UINT64:
		return uint64(C.gguf_get_val_u64(ctx, id)), true
	case C.GGUF_TYPE_INT8:
		value := int64(C.gguf_get_val_i8(ctx, id))
		return uint64(max(value, 0)), value >= 0
	case C.GGUF_TYPE_INT16:
		value := int64(C.gguf_get_val_i16(ctx, id))
		return uint64(max(value, 0)), value >= 0
	case C.GGUF_TYPE_INT32:
		value := int64(C.gguf_get_val_i32(ctx, id))
		return uint64(max(value, 0)), value >= 0
	case C.GGUF_TYPE_INT64:
		value := int64(C.gguf_get_val_i64(ctx, id))
		return uint64(max(value, 0)), value >= 0
	default:
		return 0, false
	}
}

func ggufSigned(ctx *C.struct_gguf_context, key string) (int64, bool) {
	if value, ok := ggufUnsigned(ctx, key); ok && value <= math.MaxInt64 {
		return int64(value), true
	}
	return 0, false
}
