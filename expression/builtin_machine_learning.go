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
		bf, err := newBaseBuiltinFuncWithTp(ctx, "mlapply", args, types.ETReal, types.ETString, types.ETReal, types.ETReal, types.ETReal, types.ETReal)
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

	fmt.Println(">>>>>>>>>>????????>>>>>>>>>> ", b.args, b.args[1], b.args[1].GetType())

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

	decodeDuf := bytes.NewBuffer(modelData)
	dec := gob.NewDecoder(decodeDuf)
	var weights []*tensor.Dense
	err = dec.Decode(&weights)
	if err != nil {
		return 0, false, nil
	}

	// TODO: construct the model

	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 1, 28, 28), gorgonia.WithName("x"))
	m := newConvNet(g)
	if err = m.fwd(x); err != nil {
		return 0, false, nil
	}

	learnables := m.learnables()
	for i, n := range learnables {
		if err = gorgonia.Let(n, weights[i]); err != nil {
			return 0, false, nil
		}
	}

	input := make([]float64, 28 * 28)
	val, err := b.args[1].Eval(row)
	if err != nil {
		return 0, false, nil
	}
	byts, err := val.ToBytes()
	if err != nil {
		return 0, false, nil
	}
	for i, x := range byts {
		input[i] = float64(x)
	}
	xT := tensor.New(tensor.WithShape(1,1,28,28), tensor.WithBacking(input))
	if err = gorgonia.Let(x, xT); err != nil {
		return 0, false, nil
	}

	machine := gorgonia.NewTapeMachine(g)

	if err = machine.RunAll(); err != nil {
		return 0, false, err
	}

	pred := m.out.Value().Data().([]float64)
	idx := -1
	for j := 0; j < 10; j++ {
		if idx == -1 || math.Abs(pred[j]) > math.Abs(pred[idx]) {
			idx = j
		}
	}
	return float64(idx), false, nil
}


var dt tensor.Dtype

func init() {
	dt = tensor.Float64
}

type convnet struct {
	g                  *gorgonia.ExprGraph
	w0, w1, w2, w3, w4 *gorgonia.Node // weights. the number at the back indicates which layer it's used for
	d0, d1, d2, d3     float64        // dropout probabilities

	out *gorgonia.Node
}

func newConvNet(g *gorgonia.ExprGraph) *convnet {
	w0 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128*3*3, 625), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(625, 10), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &convnet{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,

		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.55,
	}
}

func (m *convnet) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}

// This function is particularly verbose for educational reasons. In reality, you'd wrap up the layers within a layer struct type and perform per-layer activations
func (m *convnet) fwd(x *gorgonia.Node) (err error) {
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node

	// LAYER 0
	// here we convolve with stride = (1, 1) and padding = (1, 1),
	// which is your bog standard convolution for convnet
	if c0, err = gorgonia.Conv2d(x, m.w0, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		panic(err)
	}
	if a0, err = gorgonia.Rectify(c0); err != nil {
		panic(err)
	}
	if p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		panic(err)
	}
	if l0, err = gorgonia.Dropout(p0, m.d0); err != nil {
		panic(err)
	}

	// Layer 1
	if c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		panic(err)
	}
	if a1, err = gorgonia.Rectify(c1); err != nil {
		panic(err)
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		panic(err)
	}
	if l1, err = gorgonia.Dropout(p1, m.d1); err != nil {
		panic(err)
	}

	// Layer 2
	if c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		panic(err)
	}
	if a2, err = gorgonia.Rectify(c2); err != nil {
		panic(err)
	}
	if p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		panic(err)
	}

	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	if r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w}); err != nil {
		panic(err)
	}
	if l2, err = gorgonia.Dropout(r2, m.d2); err != nil {
		panic(err)
	}

	// Layer 3
	if fc, err = gorgonia.Mul(l2, m.w3); err != nil {
		panic(err)
	}
	if a3, err = gorgonia.Rectify(fc); err != nil {
		panic(err)
	}
	if l3, err = gorgonia.Dropout(a3, m.d3); err != nil {
		panic(err)
	}

	// output decode
	//var out *gorgonia.Node
	//if out, err = gorgonia.Mul(l3, m.w4); err != nil {
	//	return errors.Wrapf(err, "Unable to multiply l3 and w4")
	//}
	//m.out, err = gorgonia.SoftMax(out)
	m.out, err = gorgonia.Mul(l3, m.w4)
	return
}
