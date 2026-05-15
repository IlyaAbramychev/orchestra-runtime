package handler

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func writeRuntimeError(w http.ResponseWriter, err error) {
	payload := map[string]string{"error": err.Error()}
	if code := runtimeErrorCode(err); code != "" {
		payload["code"] = code
		payload["type"] = code
	}
	writeJSON(w, runtimeHTTPStatus(err), payload)
}

func runtimeHTTPStatus(err error) int {
	if err == nil {
		return http.StatusInternalServerError
	}
	var badReq *badRequestErr
	if errors.As(err, &badReq) {
		return http.StatusBadRequest
	}
	var contextLength *engine.ContextLengthExceededError
	if errors.As(err, &contextLength) {
		return http.StatusBadRequest
	}
	if errors.Is(err, context.Canceled) {
		return 499
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return http.StatusGatewayTimeout
	}
	if errors.Is(err, rpc.ErrWorkerCrashed) {
		return http.StatusServiceUnavailable
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "queue full"):
		return http.StatusTooManyRequests
	case strings.Contains(msg, "no model loaded"),
		strings.Contains(msg, "model not found"):
		return http.StatusNotFound
	case strings.Contains(msg, "prompt too long"),
		strings.Contains(msg, "input too long"),
		strings.Contains(msg, "context window"),
		strings.Contains(msg, "context_overflow"),
		strings.Contains(msg, "does not support"),
		strings.Contains(msg, "custom chat template failed"):
		return http.StatusBadRequest
	case strings.Contains(msg, "engine not ready"),
		strings.Contains(msg, "worker not ready"):
		return http.StatusServiceUnavailable
	default:
		return http.StatusInternalServerError
	}
}

func runtimeErrorCode(err error) string {
	if err == nil {
		return ""
	}
	var contextLength *engine.ContextLengthExceededError
	if errors.As(err, &contextLength) {
		return contextLength.Code()
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "prompt too long"),
		strings.Contains(msg, "input too long"),
		strings.Contains(msg, "context window"),
		strings.Contains(msg, "context_overflow"):
		return engine.ContextLengthExceededCode
	default:
		return ""
	}
}

func readJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func hasMeaningfulRawJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return false
	}
	if bytes.Equal(trimmed, []byte("{}")) || bytes.Equal(trimmed, []byte("[]")) {
		return false
	}
	return true
}
