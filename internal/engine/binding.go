package engine

/*
// Keep this bridge rebuilt together with the pinned llama.cpp revision: the
// C++ model registry is statically linked into the worker binary.
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include -I${SRCDIR}/../../llama.cpp/tools/mtmd
#cgo CXXFLAGS: -std=c++17 -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include -I${SRCDIR}/../../llama.cpp/common -I${SRCDIR}/../../llama.cpp/tools/mtmd -I${SRCDIR}/../../llama.cpp/vendor
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/common -L${SRCDIR}/../../llama.cpp/build/tools/mtmd -L${SRCDIR}/../../llama.cpp/build/src -L${SRCDIR}/../../llama.cpp/build/ggml/src -lllama-common -lllama-common-base -lmtmd -lllama -lggml -lggml-base -lggml-cpu -lstdc++ -lm
#cgo darwin LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/ggml/src/ggml-metal -L${SRCDIR}/../../llama.cpp/build/ggml/src/ggml-blas -lggml-metal -lggml-blas -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
#include "llama_bridge.h"
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"strings"
	"unsafe"
)

// Token is a llama token ID.
type Token = C.llama_token

// --- Backend ---

func llamaBackendInit() {
	C.llama_backend_init()
}

func llamaBackendFree() {
	C.llama_backend_free()
}

// JSONSchemaToGrammar converts a JSON Schema document to llama.cpp GBNF using
// the vendored llama.cpp converter. The returned grammar is suitable for
// llama_sampler_init_grammar.
func JSONSchemaToGrammar(schema string) (string, error) {
	cSchema := C.CString(schema)
	defer C.free(unsafe.Pointer(cSchema))

	result := C.bridge_json_schema_to_grammar(cSchema)
	defer C.bridge_schema_grammar_result_free(result)

	if result.error != nil {
		return "", fmt.Errorf("convert JSON schema to grammar: %s", C.GoString(result.error))
	}
	if result.grammar == nil {
		return "", fmt.Errorf("convert JSON schema to grammar: empty result")
	}
	return C.GoString(result.grammar), nil
}

type mtmdContext struct {
	ptr *C.mtmd_context
}

func mtmdContextLoad(path string, model *llamaModel, opts LoadOptions) (*mtmdContext, error) {
	if path == "" {
		return nil, nil
	}
	if model == nil || model.ptr == nil {
		return nil, fmt.Errorf("load mmproj: text model is not loaded")
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	params := C.mtmd_context_params_default()
	params.use_gpu = C.bool(opts.MMProjUseGPU)
	params.n_threads = C.int(opts.Threads)
	params.warmup = C.bool(false)
	switch opts.FlashAttn {
	case 0:
		params.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	case 1:
		params.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	}

	ptr := C.mtmd_init_from_file(cPath, model.ptr, params)
	if ptr == nil {
		return nil, fmt.Errorf("failed to load mmproj from %s", path)
	}
	if !bool(C.mtmd_support_vision(ptr)) {
		C.mtmd_free(ptr)
		return nil, fmt.Errorf("mmproj does not support vision input: %s", path)
	}
	return &mtmdContext{ptr: ptr}, nil
}

func (m *mtmdContext) Free() {
	if m != nil && m.ptr != nil {
		C.mtmd_free(m.ptr)
		m.ptr = nil
	}
}

func mtmdDefaultMarker() string {
	return C.GoString(C.mtmd_default_marker())
}

func mtmdEvalPrompt(mtmd *mtmdContext, lctx *llamaContext, prompt string, images [][]byte, nBatch int) (int, error) {
	if mtmd == nil || mtmd.ptr == nil {
		return 0, fmt.Errorf("multimodal images require a loaded mmproj")
	}
	if lctx == nil || lctx.ptr == nil {
		return 0, fmt.Errorf("llama context is not loaded")
	}
	if nBatch <= 0 {
		nBatch = 1
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	var cData unsafe.Pointer
	var cLens unsafe.Pointer
	var dataSlice []*C.uchar
	var lensSlice []C.size_t
	if len(images) > 0 {
		cData = C.malloc(C.size_t(len(images)) * C.size_t(unsafe.Sizeof(uintptr(0))))
		cLens = C.malloc(C.size_t(len(images)) * C.size_t(unsafe.Sizeof(C.size_t(0))))
		if cData == nil || cLens == nil {
			if cData != nil {
				C.free(cData)
			}
			if cLens != nil {
				C.free(cLens)
			}
			return 0, fmt.Errorf("allocate image pointer array")
		}
		defer C.free(cData)
		defer C.free(cLens)
		dataSlice = unsafe.Slice((**C.uchar)(cData), len(images))
		lensSlice = unsafe.Slice((*C.size_t)(cLens), len(images))
		for i, img := range images {
			if len(img) == 0 {
				return 0, fmt.Errorf("image %d is empty", i)
			}
			ptr := C.CBytes(img)
			defer C.free(ptr)
			dataSlice[i] = (*C.uchar)(ptr)
			lensSlice[i] = C.size_t(len(img))
		}
	}

	result := C.bridge_mtmd_eval_prompt(
		mtmd.ptr,
		lctx.ptr,
		cPrompt,
		(**C.uchar)(cData),
		(*C.size_t)(cLens),
		C.size_t(len(images)),
		C.bool(true),
		C.bool(true),
		C.int32_t(nBatch),
	)
	defer C.bridge_mtmd_eval_result_free(result)
	if result.error != nil {
		return int(result.n_past), fmt.Errorf("multimodal prompt eval: %s", C.GoString(result.error))
	}
	if result.code != 0 {
		return int(result.n_past), fmt.Errorf("multimodal prompt eval failed with code %d", int(result.code))
	}
	return int(result.n_past), nil
}

// --- Model ---

type llamaModel struct {
	ptr *C.struct_llama_model
}

type ModelParams struct {
	NGPULayers int
	UseMmap    bool
	// UseMlock pins the model pages in RAM so the OS can't evict them to
	// swap. Useful when you want deterministic latency; costly because it
	// reserves the full model size in physical memory regardless of idle.
	UseMlock bool
}

func llamaModelLoad(path string, params ModelParams) (*llamaModel, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	mParams := C.llama_model_default_params()
	mParams.n_gpu_layers = C.int32_t(params.NGPULayers)
	mParams.use_mmap = C.bool(params.UseMmap)
	mParams.use_mlock = C.bool(params.UseMlock)

	ptr := C.llama_model_load_from_file(cPath, mParams)
	if ptr == nil {
		return nil, fmt.Errorf("failed to load model from %s", path)
	}
	return &llamaModel{ptr: ptr}, nil
}

func (m *llamaModel) Free() {
	if m.ptr != nil {
		C.llama_model_free(m.ptr)
		m.ptr = nil
	}
}

func (m *llamaModel) Desc() string {
	buf := make([]byte, 256)
	C.llama_model_desc(m.ptr, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
	return C.GoString((*C.char)(unsafe.Pointer(&buf[0])))
}

func (m *llamaModel) Size() uint64 {
	return uint64(C.llama_model_size(m.ptr))
}

func (m *llamaModel) NParams() uint64 {
	return uint64(C.llama_model_n_params(m.ptr))
}

func (m *llamaModel) NCtxTrain() int {
	return int(C.llama_model_n_ctx_train(m.ptr))
}

// NEmbd returns the embedding dimension (hidden size). Used to size embedding
// result vectors.
func (m *llamaModel) NEmbd() int {
	return int(C.llama_model_n_embd(m.ptr))
}

// --- Vocab ---

type llamaVocab struct {
	ptr *C.struct_llama_vocab
}

func (m *llamaModel) Vocab() *llamaVocab {
	return &llamaVocab{ptr: C.llama_model_get_vocab(m.ptr)}
}

func (v *llamaVocab) NTokens() int {
	return int(C.llama_vocab_n_tokens(v.ptr))
}

func (v *llamaVocab) BOS() Token {
	return C.llama_vocab_bos(v.ptr)
}

func (v *llamaVocab) EOS() Token {
	return C.llama_vocab_eos(v.ptr)
}

func (v *llamaVocab) IsEOG(token Token) bool {
	return bool(C.llama_vocab_is_eog(v.ptr, token))
}

func (v *llamaVocab) GetAddBOS() bool {
	return bool(C.llama_vocab_get_add_bos(v.ptr))
}

func (v *llamaVocab) Tokenize(text string, addSpecial, parseSpecial bool) []Token {
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	textLen := C.int32_t(len(text))

	// First call to get token count
	n := C.llama_tokenize(v.ptr, cText, textLen, nil, 0, C.bool(addSpecial), C.bool(parseSpecial))
	if n == 0 {
		return nil
	}

	// n is negative when the buffer is too small — absolute value is the required size
	count := n
	if count < 0 {
		count = -count
	}

	tokens := make([]Token, count)
	n = C.llama_tokenize(v.ptr, cText, textLen, &tokens[0], count, C.bool(addSpecial), C.bool(parseSpecial))
	if n < 0 {
		return nil
	}
	return tokens[:n]
}

func (v *llamaVocab) TokenToStr(token Token) string {
	buf := make([]byte, 128)
	n := C.llama_token_to_piece(v.ptr, token, (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, C.bool(false))
	if n < 0 {
		return ""
	}
	return string(buf[:n])
}

// --- Context ---

type llamaContext struct {
	ptr *C.struct_llama_context
}

type ContextParams struct {
	NCtx     int
	NBatch   int
	NThreads int
	// NSeqMax — max number of distinct sequences (parallel requests). We keep
	// 1 for now; bumping this requires slot-aware inference which we haven't
	// written yet.
	NSeqMax int
	// RopeFreqBase / RopeFreqScale — 0 = inherit from GGUF metadata. Override
	// when you want to extrapolate context length (Together's YaRN, etc.)
	RopeFreqBase  float32
	RopeFreqScale float32
	// FlashAttn: -1=auto (default), 0=disabled, 1=enabled. Big perf win on
	// long contexts when the model supports it.
	FlashAttn int
	// OffloadKQV — move the KV cache itself to VRAM. On Apple Silicon unified
	// memory there's no real separation, but on discrete GPU this matters.
	OffloadKQV bool
	// TypeK / TypeV — cache element type as a string. "" means inherit (f16).
	// Supported: "f16", "f32", "q8_0", "q4_0", "q4_1", "q5_0", "q5_1".
	TypeK string
	TypeV string
}

// ggmlTypeFromString maps our string names to ggml_type enum values.
// Matches llama.cpp's --cache-type-k / --cache-type-v CLI flag semantics.
func ggmlTypeFromString(s string) (C.enum_ggml_type, bool) {
	switch s {
	case "", "f16":
		return C.GGML_TYPE_F16, true
	case "f32":
		return C.GGML_TYPE_F32, true
	case "q8_0":
		return C.GGML_TYPE_Q8_0, true
	case "q4_0":
		return C.GGML_TYPE_Q4_0, true
	case "q4_1":
		return C.GGML_TYPE_Q4_1, true
	case "q5_0":
		return C.GGML_TYPE_Q5_0, true
	case "q5_1":
		return C.GGML_TYPE_Q5_1, true
	}
	return C.GGML_TYPE_F16, false
}

func llamaNewContext(model *llamaModel, params ContextParams) (*llamaContext, error) {
	cParams := C.llama_context_default_params()
	cParams.n_ctx = C.uint32_t(params.NCtx)
	cParams.n_batch = C.uint32_t(params.NBatch)
	cParams.n_threads = C.int32_t(params.NThreads)
	cParams.n_threads_batch = C.int32_t(params.NThreads)

	if params.NSeqMax > 0 {
		cParams.n_seq_max = C.uint32_t(params.NSeqMax)
	}
	if params.RopeFreqBase > 0 {
		cParams.rope_freq_base = C.float(params.RopeFreqBase)
	}
	if params.RopeFreqScale > 0 {
		cParams.rope_freq_scale = C.float(params.RopeFreqScale)
	}
	// FlashAttn: -1 leaves default (auto); 0 or 1 explicit.
	switch params.FlashAttn {
	case 0:
		cParams.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	case 1:
		cParams.flash_attn_type = C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	}
	cParams.offload_kqv = C.bool(params.OffloadKQV)

	if t, ok := ggmlTypeFromString(params.TypeK); ok {
		cParams.type_k = t
	}
	if t, ok := ggmlTypeFromString(params.TypeV); ok {
		cParams.type_v = t
	}

	ptr := C.llama_init_from_model(model.ptr, cParams)
	if ptr == nil {
		return nil, fmt.Errorf("failed to create context")
	}
	return &llamaContext{ptr: ptr}, nil
}

func (c *llamaContext) Free() {
	if c.ptr != nil {
		C.llama_free(c.ptr)
		c.ptr = nil
	}
}

func (c *llamaContext) NCtx() int {
	return int(C.llama_n_ctx(c.ptr))
}

// ClearKVCache resets the KV cache so the context is fresh for a new conversation.
func (c *llamaContext) ClearKVCache() {
	mem := C.llama_get_memory(c.ptr)
	if mem != nil {
		C.llama_memory_clear(mem, C.bool(true))
	}
}

// SetEmbeddings toggles the context into embedding extraction mode. When true,
// `llama_decode` populates per-sequence embedding buffers instead of logits.
// We flip it only during Embed() and flip back after — chat contexts stay in
// logits mode by default.
func (c *llamaContext) SetEmbeddings(on bool) {
	C.llama_set_embeddings(c.ptr, C.bool(on))
}

// EmbeddingsSeq returns the pooled embedding for sequence `seqID`. Valid only
// after a `decode` with an embedding-mode batch. Caller must copy out before
// running another decode — the underlying buffer is owned by llama.cpp.
func (c *llamaContext) EmbeddingsSeq(seqID int, nEmbd int) []float32 {
	ptr := C.llama_get_embeddings_seq(c.ptr, C.llama_seq_id(seqID))
	if ptr == nil {
		return nil
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	out := make([]float32, nEmbd)
	copy(out, src)
	return out
}

// EmbeddingsAll returns the embedding for a non-pooled context (pooling_type=NONE)
// at position `i`. Used when the model doesn't auto-pool.
func (c *llamaContext) EmbeddingsIth(i int, nEmbd int) []float32 {
	ptr := C.llama_get_embeddings_ith(c.ptr, C.int32_t(i))
	if ptr == nil {
		return nil
	}
	src := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	out := make([]float32, nEmbd)
	copy(out, src)
	return out
}

// --- Batch ---

type llamaBatch struct {
	b        C.struct_llama_batch
	capacity int
}

func llamaBatchInit(nTokens, nSeqMax int) *llamaBatch {
	return &llamaBatch{
		b:        C.llama_batch_init(C.int32_t(nTokens), 0, C.int32_t(nSeqMax)),
		capacity: nTokens,
	}
}

func (b *llamaBatch) Free() {
	C.llama_batch_free(b.b)
}

func (b *llamaBatch) Clear() {
	C.bridge_batch_clear(&b.b)
}

func (b *llamaBatch) Add(token Token, pos int, seqID int, logits bool) error {
	if b.NTokens() >= b.capacity {
		return fmt.Errorf("llama batch capacity exceeded: %d >= %d", b.NTokens(), b.capacity)
	}
	C.bridge_batch_add(&b.b, token, C.llama_pos(pos), C.llama_seq_id(seqID), C.bool(logits))
	return nil
}

func (b *llamaBatch) NTokens() int {
	return int(b.b.n_tokens)
}

// --- Decode ---

func llamaDecode(ctx *llamaContext, batch *llamaBatch) error {
	ret := C.llama_decode(ctx.ptr, batch.b)
	if ret < 0 {
		return fmt.Errorf("llama_decode failed with code %d", int(ret))
	}
	return nil
}

// --- Sampler ---

type llamaSampler struct {
	ptr    *C.struct_llama_sampler
	common *C.struct_common_sampler
}

// SamplerOpts bundles all knobs exposed in the public sampling panel. Nil
// fields fall back to their “disabled” value (e.g. top_k=0 means no truncate).
type SamplerOpts struct {
	Temp             float32
	TopK             int
	TopP             float32
	MinP             float32
	TypicalP         float32
	RepeatPenalty    float32
	RepeatLastN      int
	FrequencyPenalty float32
	PresencePenalty  float32
	Mirostat         int // 0 off, 1 v1, 2 v2
	MirostatTau      float32
	MirostatEta      float32
	Seed             uint32
	NVocab           int32 // required for penalty samplers
	Vocab            *llamaVocab
	Model            *llamaModel
	Grammar          string
	GrammarLazy      bool
	GrammarTriggers  []GrammarTrigger
	GenerationPrompt string
}

// NewSamplerChain composes the full llama.cpp sampler chain. Order matters:
// repetition penalties → grammar constraints → truncation
// (top_k/top_p/min_p/typical) → temperature → mirostat (or dist). See
// llama.cpp/examples/main/main.cpp for reference.
func NewSamplerChain(o SamplerOpts) (*llamaSampler, error) {
	if strings.TrimSpace(o.Grammar) != "" && (o.GenerationPrompt != "" || o.GrammarLazy || len(o.GrammarTriggers) > 0) {
		return newCommonNativeSampler(o)
	}
	chainParams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(chainParams)

	// Repetition controls: only add if any is non-default — cheap but not free.
	if (o.RepeatPenalty > 0 && o.RepeatPenalty != 1.0) || o.FrequencyPenalty != 0 || o.PresencePenalty != 0 {
		last := o.RepeatLastN
		if last == 0 {
			last = 64
		}
		rp := o.RepeatPenalty
		if rp == 0 {
			rp = 1.0
		}
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_penalties(
			C.int32_t(last),
			C.float(rp),
			C.float(o.FrequencyPenalty),
			C.float(o.PresencePenalty),
		))
	}

	if grammar := strings.TrimSpace(o.Grammar); grammar != "" {
		if o.Vocab == nil || o.Vocab.ptr == nil {
			C.llama_sampler_free(chain)
			return nil, fmt.Errorf("grammar-constrained decoding requires a loaded vocab")
		}
		cGrammar := C.CString(grammar)
		var grammarSampler *C.struct_llama_sampler
		if o.GrammarLazy || len(o.GrammarTriggers) > 0 || o.GenerationPrompt != "" {
			triggerJSON, marshalErr := json.Marshal(o.GrammarTriggers)
			if marshalErr != nil {
				C.free(unsafe.Pointer(cGrammar))
				C.llama_sampler_free(chain)
				return nil, fmt.Errorf("encode native grammar triggers: %w", marshalErr)
			}
			cTriggers := C.CString(string(triggerJSON))
			cGenerationPrompt := C.CString(o.GenerationPrompt)
			var cErr *C.char
			grammarSampler = C.bridge_chat_grammar_sampler_init(
				o.Vocab.ptr, cGrammar, C.bool(o.GrammarLazy), cTriggers, cGenerationPrompt, &cErr,
			)
			C.free(unsafe.Pointer(cTriggers))
			C.free(unsafe.Pointer(cGenerationPrompt))
			if cErr != nil {
				message := C.GoString(cErr)
				C.bridge_string_free(cErr)
				C.free(unsafe.Pointer(cGrammar))
				C.llama_sampler_free(chain)
				return nil, fmt.Errorf("invalid native chat grammar: %s", message)
			}
		} else {
			cRoot := C.CString("root")
			grammarSampler = C.llama_sampler_init_grammar(o.Vocab.ptr, cGrammar, cRoot)
			C.free(unsafe.Pointer(cRoot))
		}
		C.free(unsafe.Pointer(cGrammar))
		if grammarSampler == nil {
			C.llama_sampler_free(chain)
			return nil, fmt.Errorf("invalid grammar for constrained decoding")
		}
		C.llama_sampler_chain_add(chain, grammarSampler)
	}

	// Mirostat replaces top_k/top_p entirely — skip truncation if enabled.
	switch o.Mirostat {
	case 1:
		tau := o.MirostatTau
		if tau == 0 {
			tau = 5.0
		}
		eta := o.MirostatEta
		if eta == 0 {
			eta = 0.1
		}
		C.llama_sampler_chain_add(chain,
			C.llama_sampler_init_mirostat(C.int32_t(o.NVocab), C.uint32_t(o.Seed), C.float(tau), C.float(eta), 100))
	case 2:
		tau := o.MirostatTau
		if tau == 0 {
			tau = 5.0
		}
		eta := o.MirostatEta
		if eta == 0 {
			eta = 0.1
		}
		C.llama_sampler_chain_add(chain,
			C.llama_sampler_init_mirostat_v2(C.uint32_t(o.Seed), C.float(tau), C.float(eta)))
	default:
		// Standard truncation stack.
		if o.TopK > 0 {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(C.int32_t(o.TopK)))
		}
		if o.TypicalP > 0 && o.TypicalP < 1.0 {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_typical(C.float(o.TypicalP), 1))
		}
		if o.TopP > 0 && o.TopP < 1.0 {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(C.float(o.TopP), 1))
		}
		if o.MinP > 0 {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_min_p(C.float(o.MinP), 1))
		}
		if o.Temp <= 0 {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
		} else {
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(o.Temp)))
			C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(o.Seed)))
		}
	}

	return &llamaSampler{ptr: chain}, nil
}

func newCommonNativeSampler(o SamplerOpts) (*llamaSampler, error) {
	if o.Model == nil || o.Model.ptr == nil {
		return nil, fmt.Errorf("native chat sampling requires a loaded model")
	}
	options, err := json.Marshal(map[string]any{
		"seed":              o.Seed,
		"temperature":       o.Temp,
		"top_k":             o.TopK,
		"top_p":             o.TopP,
		"min_p":             o.MinP,
		"typical_p":         o.TypicalP,
		"repeat_penalty":    o.RepeatPenalty,
		"repeat_last_n":     o.RepeatLastN,
		"frequency_penalty": o.FrequencyPenalty,
		"presence_penalty":  o.PresencePenalty,
		"mirostat":          o.Mirostat,
		"mirostat_tau":      o.MirostatTau,
		"mirostat_eta":      o.MirostatEta,
	})
	if err != nil {
		return nil, err
	}
	triggers, err := json.Marshal(o.GrammarTriggers)
	if err != nil {
		return nil, err
	}
	cOptions := C.CString(string(options))
	cGrammar := C.CString(o.Grammar)
	cTriggers := C.CString(string(triggers))
	cGenerationPrompt := C.CString(o.GenerationPrompt)
	defer C.free(unsafe.Pointer(cOptions))
	defer C.free(unsafe.Pointer(cGrammar))
	defer C.free(unsafe.Pointer(cTriggers))
	defer C.free(unsafe.Pointer(cGenerationPrompt))
	var cErr *C.char
	common := C.bridge_common_sampler_init(
		o.Model.ptr, cOptions, cGrammar, C.bool(o.GrammarLazy), cTriggers, cGenerationPrompt, &cErr,
	)
	if cErr != nil {
		message := C.GoString(cErr)
		C.bridge_string_free(cErr)
		return nil, fmt.Errorf("initialize native llama.cpp sampler: %s", message)
	}
	if common == nil {
		return nil, fmt.Errorf("initialize native llama.cpp sampler: empty result")
	}
	return &llamaSampler{common: common}, nil
}

// NewGreedySampler creates a greedy (argmax) sampler.
func NewGreedySampler() *llamaSampler {
	chainParams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(chainParams)
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
	return &llamaSampler{ptr: chain}
}

func (s *llamaSampler) Sample(ctx *llamaContext, idx int) Token {
	if s.common != nil {
		return C.bridge_common_sampler_sample(s.common, ctx.ptr, C.int32_t(idx))
	}
	return C.llama_sampler_sample(s.ptr, ctx.ptr, C.int32_t(idx))
}

func (s *llamaSampler) Accept(token Token) {
	if s.common != nil {
		C.bridge_common_sampler_accept(s.common, token)
		return
	}
	C.llama_sampler_accept(s.ptr, token)
}

func (s *llamaSampler) Free() {
	if s.common != nil {
		C.bridge_common_sampler_free(s.common)
		s.common = nil
	}
	if s.ptr != nil {
		C.llama_sampler_free(s.ptr)
		s.ptr = nil
	}
}

// --- Chat template ---

type ChatMessage struct {
	Role       string
	Content    string
	Reasoning  string
	ToolName   string
	ToolCallID string
	ToolCalls  []ToolCall
	Parts      []ContentPart
	Images     []string
}

// ToolCall keeps arguments as JSON so the native llama.cpp chat layer sees
// exactly the structured history supplied by the client.
type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}

type ContentPart struct {
	Type        string
	Text        string
	ImageURL    string
	ImageDetail string
}

type GrammarTrigger struct {
	Type  int    `json:"type"`
	Value string `json:"value"`
	Token int32  `json:"token"`
}

type NativeChatRender struct {
	Prompt           string
	Grammar          string
	Parser           string
	GenerationPrompt string
	AdditionalStops  []string
	GrammarTriggers  []GrammarTrigger
	Capabilities     map[string]bool
	Format           int
	GrammarLazy      bool
	SupportsThinking bool
}

func RenderNativeChat(model *llamaModel, tmpl, messagesJSON, toolsJSON string, toolChoice int, parallelToolCalls, enableThinking bool) (*NativeChatRender, error) {
	if model == nil || model.ptr == nil {
		return nil, fmt.Errorf("native chat rendering requires a loaded model")
	}
	cTemplate := C.CString(tmpl)
	cMessages := C.CString(messagesJSON)
	cTools := C.CString(toolsJSON)
	defer C.free(unsafe.Pointer(cTemplate))
	defer C.free(unsafe.Pointer(cMessages))
	defer C.free(unsafe.Pointer(cTools))

	result := C.bridge_chat_render_native(
		model.ptr,
		cTemplate,
		cMessages,
		cTools,
		C.int32_t(toolChoice),
		C.bool(parallelToolCalls),
		C.bool(enableThinking),
	)
	defer C.bridge_chat_render_result_free(result)
	if result.error != nil {
		return nil, fmt.Errorf("native chat template: %s", C.GoString(result.error))
	}
	render := &NativeChatRender{
		Prompt:           C.GoString(result.prompt),
		Grammar:          C.GoString(result.grammar),
		Parser:           C.GoString(result.parser),
		GenerationPrompt: C.GoString(result.generation_prompt),
		Format:           int(result.format),
		GrammarLazy:      bool(result.grammar_lazy),
		SupportsThinking: bool(result.supports_thinking),
	}
	if result.additional_stops_json != nil {
		_ = json.Unmarshal([]byte(C.GoString(result.additional_stops_json)), &render.AdditionalStops)
	}
	if result.grammar_triggers_json != nil {
		_ = json.Unmarshal([]byte(C.GoString(result.grammar_triggers_json)), &render.GrammarTriggers)
	}
	if result.capabilities_json != nil {
		_ = json.Unmarshal([]byte(C.GoString(result.capabilities_json)), &render.Capabilities)
	}
	return render, nil
}

func ParseNativeChat(response string, render *NativeChatRender) (json.RawMessage, error) {
	if render == nil {
		return nil, fmt.Errorf("native chat parser parameters are required")
	}
	cResponse := C.CString(response)
	cParser := C.CString(render.Parser)
	cGenerationPrompt := C.CString(render.GenerationPrompt)
	defer C.free(unsafe.Pointer(cResponse))
	defer C.free(unsafe.Pointer(cParser))
	defer C.free(unsafe.Pointer(cGenerationPrompt))

	result := C.bridge_chat_parse_native(cResponse, cParser, cGenerationPrompt, C.int32_t(render.Format))
	defer C.bridge_chat_parse_result_free(result)
	if result.error != nil {
		return nil, fmt.Errorf("native chat parse: %s", C.GoString(result.error))
	}
	if result.message_json == nil {
		return nil, fmt.Errorf("native chat parse returned no message")
	}
	return json.RawMessage(C.GoString(result.message_json)), nil
}

func ApplyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("no messages")
	}

	cMsgs := make([]C.struct_llama_chat_message, len(messages))
	cStrings := make([]*C.char, len(messages)*2) // keep alive

	for i, msg := range messages {
		cRole := C.CString(msg.Role)
		cContent := C.CString(msg.Content)
		cStrings[i*2] = cRole
		cStrings[i*2+1] = cContent

		cMsgs[i].role = cRole
		cMsgs[i].content = cContent
	}
	defer func() {
		for _, cs := range cStrings {
			C.free(unsafe.Pointer(cs))
		}
	}()

	var cTmpl *C.char
	if tmpl != "" {
		cTmpl = C.CString(tmpl)
		defer C.free(unsafe.Pointer(cTmpl))
	}

	var outLen C.int32_t
	result := C.bridge_chat_apply_template(
		cTmpl,
		&cMsgs[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		&outLen,
	)

	if result == nil || outLen < 0 {
		return "", fmt.Errorf("failed to apply chat template (code: %d)", int(outLen))
	}
	defer C.free(unsafe.Pointer(result))

	return C.GoStringN(result, outLen), nil
}
