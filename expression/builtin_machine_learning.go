package expression

import (
	"fmt"

	"github.com/pingcap/tidb/dumpling/context"
	"github.com/pingcap/tidb/sessionctx"
	"github.com/pingcap/tidb/types"
	"github.com/pingcap/tidb/util/chunk"
	"github.com/pingcap/tidb/util/sqlexec"
)

type mlApplyFunctionClass struct {
	baseFunctionClass
}

func (c *mlApplyFunctionClass) getFunction(ctx sessionctx.Context, args []Expression) (builtinFunc, error) {
	if err := c.verifyArgs(args); err != nil {
		return nil, c.verifyArgs(args)
	}

	bf, err := newBaseBuiltinFuncWithTp(ctx, "mlapply", args, types.ETReal, types.ETString, types.ETString)
	if err != nil {
		return nil, err
	}
	bf.HasCoercibility()

	return &builtinMLApplySig{bf, false}, nil
}

type builtinMLApplySig struct {
	baseBuiltinFunc

	inited bool
}

func (b *builtinMLApplySig) Clone() builtinFunc {
	newSig := &builtinMLApplySig{}
	newSig.cloneFrom(&b.baseBuiltinFunc)
	return newSig
}

// evalInt evals ABS(value).
// See https://dev.mysql.com/doc/refman/5.7/en/mathematical-functions.html#function_abs
func (b *builtinMLApplySig) evalReal(row chunk.Row) (float64, bool, error) {
	if !b.inited {
		modelName, _, err := b.args[0].EvalString(b.ctx, row)
		if err != nil {
			return 0, false, err
		}

		// read information about this model
		exec := b.ctx.(sqlexec.RestrictedSQLExecutor)
		stmt, err := exec.ParseWithParamsInternal(context.Background(), "select model_data from mysql.ml_models where name = %?", modelName)
		if err != nil {
			return 0, false, err
		}
		rows, _, err := exec.ExecRestrictedStmt(context.Background(), stmt)
		if err != nil {
			return 0, false, err
		}
		if len(rows) == 0 {
			return 0, false, fmt.Errorf("model %v not found", modelName)
		}
		modelData := rows[0].GetBytes(0)
		fmt.Println(">>>>>>>>>>>>>> ", len(modelData))

		// TODO: construct the model

		b.inited = true
	}

	data, _, err := b.args[1].EvalString(b.ctx, row)
	if err != nil {
		return 0, false, err
	}
	fmt.Println(">>>>>>>>>>>>>> ", len(data))
	// TODO: convert data to X

	return 0, false, nil
}
