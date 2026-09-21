package engine

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"testing/iotest"
)

type nativeLogRecord struct {
	level slog.Level
	msg   string
}

func decodeNativeLog(t *testing.T, raw string) []nativeLogRecord {
	t.Helper()
	var got []nativeLogRecord
	// OneByteReader splits every record boundary across reads.
	pumpNativeLog(iotest.OneByteReader(strings.NewReader(raw)), func(level slog.Level, msg string) {
		got = append(got, nativeLogRecord{level: level, msg: msg})
	})
	return got
}

func nativeRecord(level int, text string) string {
	return string([]byte{nativeLogRecordSep, byte('0' + level)}) + text
}

func TestPumpNativeLogSplitsLinesAndMapsLevels(t *testing.T) {
	raw := nativeRecord(ggmlLogInfo, "load_tensors: offloaded 33/33 layers\nsecond line\n") +
		nativeRecord(ggmlLogWarn, "kv cache near limit\n") +
		nativeRecord(ggmlLogError, "llama_model_load: error loading model\n")

	got := decodeNativeLog(t, raw)
	want := []nativeLogRecord{
		{slog.LevelDebug, "load_tensors: offloaded 33/33 layers"},
		{slog.LevelDebug, "second line"},
		{slog.LevelWarn, "kv cache near limit"},
		{slog.LevelError, "llama_model_load: error loading model"},
	}
	if len(got) != len(want) {
		t.Fatalf("records = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("record %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestPumpNativeLogJoinsContinuationRecords(t *testing.T) {
	raw := nativeRecord(ggmlLogWarn, "loading") +
		nativeRecord(ggmlLogCont, ".") +
		nativeRecord(ggmlLogCont, ".") +
		nativeRecord(ggmlLogCont, " done\n")

	got := decodeNativeLog(t, raw)
	if len(got) != 1 || got[0].msg != "loading.. done" || got[0].level != slog.LevelWarn {
		t.Fatalf("records = %+v", got)
	}
}

func TestPumpNativeLogFlushesUnterminatedLines(t *testing.T) {
	raw := nativeRecord(ggmlLogError, "first without newline") +
		nativeRecord(ggmlLogInfo, "tail without newline")

	got := decodeNativeLog(t, raw)
	if len(got) != 2 || got[0].msg != "first without newline" || got[1].msg != "tail without newline" {
		t.Fatalf("records = %+v", got)
	}
}

func TestPumpNativeLogBoundsRunawayLines(t *testing.T) {
	text := strings.Repeat("x", 3*nativeLogMaxLine)
	raw := nativeRecord(ggmlLogInfo, text)

	var got []nativeLogRecord
	pumpNativeLog(bytes.NewReader([]byte(raw)), func(level slog.Level, msg string) {
		got = append(got, nativeLogRecord{level: level, msg: msg})
	})
	if len(got) < 2 {
		t.Fatalf("expected a line without newline to be split, got %d records", len(got))
	}
	total := 0
	for _, record := range got {
		// A flush happens once the line reaches the limit, so a record can
		// exceed it by at most one read.
		if len(record.msg) > nativeLogMaxLine+16*1024 {
			t.Fatalf("record length %d exceeds bound", len(record.msg))
		}
		total += len(record.msg)
	}
	if total != len(text) {
		t.Fatalf("decoded %d bytes, want %d", total, len(text))
	}
}
