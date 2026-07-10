package supervisor

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/operium/orchestra-runtime/internal/engine"
	"github.com/operium/orchestra-runtime/internal/rpc"
)

func TestToRPCParamsPreservesChatTemplate(t *testing.T) {
	got := toRPCParams(engine.CompletionParams{ChatTemplate: "{{ custom_template }}"})
	if got.ChatTemplate != "{{ custom_template }}" {
		t.Fatalf("chat template lost in host RPC conversion: %q", got.ChatTemplate)
	}
}

func TestValidateWorkerHandshakeRejectsMixedProtocolVersions(t *testing.T) {
	err := validateWorkerHandshake(rpc.PingResult{
		Pong:            true,
		ProtocolVersion: rpc.ProtocolVersion - 1,
		Version:         "0.3.3",
		BuildCommit:     "old-worker",
	}, "0.4.0", "new-host")
	if err == nil || !strings.Contains(err.Error(), "same release") {
		t.Fatalf("expected actionable protocol mismatch, got %v", err)
	}
	if err := validateWorkerHandshake(rpc.PingResult{Pong: true, ProtocolVersion: rpc.ProtocolVersion, Version: "0.4.0", BuildCommit: "same"}, "0.4.0", "same"); err != nil {
		t.Fatalf("matching protocol was rejected: %v", err)
	}
}

func TestValidateWorkerHandshakeRejectsDifferentBuildAtSameProtocol(t *testing.T) {
	err := validateWorkerHandshake(rpc.PingResult{
		Pong: true, ProtocolVersion: rpc.ProtocolVersion, Version: "0.4.0", BuildCommit: "worker-build",
	}, "0.4.0", "host-build")
	if err == nil || !strings.Contains(err.Error(), "version host=0.4.0 worker=0.4.0") {
		t.Fatalf("expected build mismatch, got %v", err)
	}
}

func TestToRPCMessagesPreservesMultimodalParts(t *testing.T) {
	got := toRPCMessages([]engine.ChatMessage{{
		Role: "user",
		Parts: []engine.ContentPart{{
			Type:        "image_url",
			ImageURL:    "data:image/png;base64,aGVsbG8=",
			ImageDetail: "high",
		}},
	}})
	if len(got) != 1 || len(got[0].Parts) != 1 {
		t.Fatalf("parts lost in RPC conversion: %+v", got)
	}
	if got[0].Parts[0].ImageDetail != "high" {
		t.Fatalf("image detail lost: %+v", got[0].Parts[0])
	}
}

func TestCallStreamSendsCancelOnce(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()

	w := NewWorker(Options{})
	w.conn = client
	w.codec = rpc.NewCodec(client)
	w.ready.Store(true)
	go w.readLoop()

	serverCodec := rpc.NewCodec(server)
	reqCh := make(chan readResult, 1)
	go func() {
		env, err := serverCodec.Read()
		reqCh <- readResult{env: env, err: err}
	}()

	ctx, cancel := context.WithCancel(context.Background())
	frames, err := w.CallStream(ctx, rpc.MethodCompleteStream, rpc.CompleteParams{})
	if err != nil {
		t.Fatalf("CallStream failed: %v", err)
	}

	var req *rpc.Envelope
	select {
	case res := <-reqCh:
		if res.err != nil {
			t.Fatalf("read initial request: %v", res.err)
		}
		req = res.env
	case <-time.After(time.Second):
		t.Fatal("initial request was not written")
	}
	if req.Method != rpc.MethodCompleteStream {
		t.Fatalf("unexpected method %q", req.Method)
	}

	cancel()

	cancelCount := 0
	deadline := time.Now().Add(100 * time.Millisecond)
	for time.Now().Before(deadline) {
		_ = server.SetReadDeadline(time.Now().Add(20 * time.Millisecond))
		env, err := serverCodec.Read()
		if err != nil {
			if isTimeout(err) {
				break
			}
			t.Fatalf("read cancel frame: %v", err)
		}
		if env.Method == rpc.MethodCancel {
			cancelCount++
		}
	}
	_ = server.SetReadDeadline(time.Time{})

	if cancelCount != 1 {
		t.Fatalf("expected exactly one cancel frame, got %d", cancelCount)
	}

	payload, _ := json.Marshal(rpc.StreamChunk{Done: true, FinishReason: "stop"})
	if err := serverCodec.Write(&rpc.Envelope{ID: req.ID, Kind: rpc.KindFinal, Result: payload}); err != nil {
		t.Fatalf("write final: %v", err)
	}

	select {
	case _, ok := <-frames:
		if ok {
			t.Fatal("expected stream channel to close after cancelled final")
		}
	case <-time.After(time.Second):
		t.Fatal("stream channel did not close")
	}
}

func TestHandleDisconnectClosesPendingCalls(t *testing.T) {
	w := NewWorker(Options{})
	w.ready.Store(true)

	call := &pendingCall{final: make(chan *rpc.Envelope, 1)}
	stream := &pendingCall{
		final:  make(chan *rpc.Envelope, 1),
		chunks: make(chan *rpc.Envelope, 1),
	}
	w.pending["call"] = call
	w.pending["stream"] = stream

	w.handleDisconnect(io.EOF)

	if w.ready.Load() {
		t.Fatal("worker should not remain ready after disconnect")
	}
	assertClosed(t, call.final, "call final")
	assertClosed(t, stream.final, "stream final")
	assertClosed(t, stream.chunks, "stream chunks")
}

func isTimeout(err error) bool {
	var netErr net.Error
	return errors.As(err, &netErr) && netErr.Timeout()
}

type readResult struct {
	env *rpc.Envelope
	err error
}

func assertClosed[T any](t *testing.T, ch <-chan T, name string) {
	t.Helper()
	select {
	case _, ok := <-ch:
		if ok {
			t.Fatalf("%s channel is not closed", name)
		}
	case <-time.After(time.Second):
		t.Fatalf("%s channel did not close", name)
	}
}
