package expression

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math"
	"strings"

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

	modelName, _, err := args[0].EvalString(ctx, chunk.Row{})
	if err != nil {
		return nil, err
	}
	if strings.Contains(strings.ToLower(modelName), "iris") {
		bf, err := newBaseBuiltinFuncWithTp(ctx, "mlapply", args, types.ETReal, types.ETString, types.ETString, types.ETReal, types.ETReal, types.ETReal)
		if err != nil {
			return nil, nil
		}
		return &builtinMLApply4IrisSig{bf}, nil
	}

	bf, err := newBaseBuiltinFuncWithTp(ctx, "mlapply", args, types.ETReal, types.ETString, types.ETString)
	if err != nil {
		return nil, err
	}

	return &builtinMLApplySig{bf, false}, nil
}

type builtinMLApply4IrisSig struct {
	baseBuiltinFunc
}

func (b *builtinMLApply4IrisSig) evalReal(row chunk.Row) (float64, bool, error) {
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

	decodeDuf := bytes.NewBuffer(modelData)
	dec := gob.NewDecoder(decodeDuf)
	var thetaT *tensor.Dense
	err = dec.Decode(&thetaT)
	if err != nil {
		return 0, false, nil
	}

	g := gorgonia.NewGraph()
	theta := gorgonia.NodeFromAny(g, thetaT, gorgonia.WithName("theta"))
	values := make([]float64, 4)
	xT := tensor.New(tensor.WithBacking(values))
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	y, err := gorgonia.Mul(x, theta)

	machine := gorgonia.NewTapeMachine(g)

	values[0], _, err = b.args[1].EvalReal(b.ctx, row)
	if err != nil {
		return 0, false, err
	}
	values[1], _, err = b.args[2].EvalReal(b.ctx, row)
	if err != nil {
		return 0, false, err
	}
	values[2], _, err = b.args[3].EvalReal(b.ctx, row)
	if err != nil {
		return 0, false, err
	}
	values[3], _, err = b.args[4].EvalReal(b.ctx, row)
	if err != nil {
		return 0, false, err
	}

	if err = machine.RunAll(); err != nil {
		return 0, false, err
	}

	result := math.Round(y.Value().Data().(float64))
	machine.Reset()

	return result, false, nil
}

func (b *builtinMLApply4IrisSig) Clone() builtinFunc {
	newSig := &builtinMLApply4IrisSig{}
	newSig.cloneFrom(&b.baseBuiltinFunc)
	return newSig
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
