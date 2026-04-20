package engine

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include
#cgo LDFLAGS: -lllama -lggml -lstdc++ -lm
#cgo darwin LDFLAGS: -framework Accelerate -framework Metal -framework MetalKit -framework Foundation
#include "llama_bridge.h"
#include <stdlib.h>
#include <stdbool.h>
*/
import "C"
import (
	"fmt"
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

// --- Model ---

type llamaModel struct {
	ptr *C.struct_llama_model
}

type ModelParams struct {
	NGPULayers int
	UseMmap    bool
}

func llamaModelLoad(path string, params ModelParams) (*llamaModel, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	mParams := C.llama_model_default_params()
	mParams.n_gpu_layers = C.int32_t(params.NGPULayers)
	mParams.use_mmap = C.bool(params.UseMmap)

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
	NCtx      int
	NBatch    int
	NThreads  int
}

func llamaNewContext(model *llamaModel, params ContextParams) (*llamaContext, error) {
	cParams := C.llama_context_default_params()
	cParams.n_ctx = C.uint32_t(params.NCtx)
	cParams.n_batch = C.uint32_t(params.NBatch)
	cParams.n_threads = C.int32_t(params.NThreads)
	cParams.n_threads_batch = C.int32_t(params.NThreads)

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

// --- Batch ---

type llamaBatch struct {
	b C.struct_llama_batch
}

func llamaBatchInit(nTokens, nSeqMax int) *llamaBatch {
	return &llamaBatch{
		b: C.llama_batch_init(C.int32_t(nTokens), 0, C.int32_t(nSeqMax)),
	}
}

func (b *llamaBatch) Free() {
	C.llama_batch_free(b.b)
}

func (b *llamaBatch) Clear() {
	C.bridge_batch_clear(&b.b)
}

func (b *llamaBatch) Add(token Token, pos int, seqID int, logits bool) {
	C.bridge_batch_add(&b.b, token, C.llama_pos(pos), C.llama_seq_id(seqID), C.bool(logits))
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

// --- Logits ---

func llamaGetLogitsIth(ctx *llamaContext, i int) []float32 {
	model := C.llama_get_model(ctx.ptr)
	vocab := C.llama_model_get_vocab(model)
	nVocab := int(C.llama_vocab_n_tokens(vocab))

	logitsPtr := C.llama_get_logits_ith(ctx.ptr, C.int32_t(i))
	if logitsPtr == nil {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(logitsPtr)), nVocab)
}

// --- Sampler ---

type llamaSampler struct {
	ptr *C.struct_llama_sampler
}

// NewSamplerChain creates a sampler chain with temperature, top-k, top-p, and dist sampling.
func NewSamplerChain(temp float32, topK int, topP float32, seed uint32) *llamaSampler {
	chainParams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(chainParams)

	if topK > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_k(C.int32_t(topK)))
	}
	if topP > 0 && topP < 1.0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_top_p(C.float(topP), 1))
	}
	if temp > 0 {
		C.llama_sampler_chain_add(chain, C.llama_sampler_init_temp(C.float(temp)))
	}
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_dist(C.uint32_t(seed)))

	return &llamaSampler{ptr: chain}
}

// NewGreedySampler creates a greedy (argmax) sampler.
func NewGreedySampler() *llamaSampler {
	chainParams := C.llama_sampler_chain_default_params()
	chain := C.llama_sampler_chain_init(chainParams)
	C.llama_sampler_chain_add(chain, C.llama_sampler_init_greedy())
	return &llamaSampler{ptr: chain}
}

func (s *llamaSampler) Sample(ctx *llamaContext, idx int) Token {
	return C.llama_sampler_sample(s.ptr, ctx.ptr, C.int32_t(idx))
}

func (s *llamaSampler) Accept(token Token) {
	C.llama_sampler_accept(s.ptr, token)
}

func (s *llamaSampler) Free() {
	if s.ptr != nil {
		C.llama_sampler_free(s.ptr)
		s.ptr = nil
	}
}

// --- Chat template ---

type ChatMessage struct {
	Role    string
	Content string
}

func ApplyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
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
