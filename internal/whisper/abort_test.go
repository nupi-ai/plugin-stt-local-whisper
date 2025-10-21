//go:build cgo

package whisper

import (
	"context"
	"runtime/cgo"
	"testing"
	"unsafe"
)

func TestContextFromHandle(t *testing.T) {
	ctx := context.Background()
	handle := cgo.NewHandle(ctx)
	defer handle.Delete()

	got, ok := contextFromHandle(unsafe.Pointer(&handle))
	if !ok || got != ctx {
		t.Fatalf("expected original context, got %#v (ok=%v)", got, ok)
	}
}

func TestContextFromHandleCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	handle := cgo.NewHandle(ctx)
	defer handle.Delete()

	got, ok := contextFromHandle(unsafe.Pointer(&handle))
	if !ok {
		t.Fatalf("expected context, got ok=false")
	}
	select {
	case <-got.Done():
	default:
		t.Fatalf("expected cancelled context")
	}
}

func TestContextFromHandleInvalidType(t *testing.T) {
	handle := cgo.NewHandle("not a context")
	defer handle.Delete()

	if _, ok := contextFromHandle(unsafe.Pointer(&handle)); ok {
		t.Fatalf("expected ok=false for non-context handle")
	}
}

func TestContextFromHandleDeleted(t *testing.T) {
	ctx := context.Background()
	handle := cgo.NewHandle(ctx)
	ptr := unsafe.Pointer(&handle)
	handle.Delete()

	if _, ok := contextFromHandle(ptr); ok {
		t.Fatalf("expected ok=false for deleted handle")
	}
}

func TestContextFromHandleNil(t *testing.T) {
	if _, ok := contextFromHandle(nil); ok {
		t.Fatalf("expected ok=false for nil")
	}
}

func TestShouldAbort(t *testing.T) {
	ctx := context.Background()
	handle := cgo.NewHandle(ctx)
	defer handle.Delete()

	if shouldAbort(unsafe.Pointer(&handle)) {
		t.Fatalf("expected no abort for active context")
	}

	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel()
	cancelHandle := cgo.NewHandle(cancelCtx)
	defer cancelHandle.Delete()

	if !shouldAbort(unsafe.Pointer(&cancelHandle)) {
		t.Fatalf("expected abort for cancelled context")
	}
}
