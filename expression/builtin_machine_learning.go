package expression

import (
	"errors"
	"github.com/pingcap/tidb/sessionctx"
)

type mlApplyFunctionClass struct {
	baseFunctionClass
}

func (c *mlApplyFunctionClass) getFunction(ctx sessionctx.Context, args []Expression) (builtinFunc, error) {
	return nil, errors.New("unsupportted")
}
