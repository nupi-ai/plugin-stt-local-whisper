//go:build cgo

package engine

import (
	"context"
	"runtime/cgo"
	"unsafe"
)

func contextFromHandle(userData unsafe.Pointer) (context.Context, bool) {
	if userData == nil {
		return nil, false
	}

	handlePtr := (*cgo.Handle)(userData)
	if handlePtr == nil {
		return nil, false
	}
	handle := *handlePtr
	if handle == 0 {
		return nil, false
	}
	var (
		value     any
		recovered bool
	)

	func() {
		defer func() {
			if r := recover(); r != nil {
				recovered = true
				value = nil
			}
		}()
		value = handle.Value()
	}()

	if recovered || value == nil {
		return nil, false
	}

	ctx, ok := value.(context.Context)
	return ctx, ok
}

func shouldAbort(userData unsafe.Pointer) bool {
	ctx, ok := contextFromHandle(userData)
	if !ok {
		return false
	}
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}
