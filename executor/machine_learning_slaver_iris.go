package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"fmt"
	"os"

	"github.com/pingcap/tidb/util/sqlexec"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func (h *CoprocessorDAGHandler) HandleSlaverTraining4Iris(mlReq MLModelReq) ([]byte, error) {
	// read data
	xT, yT, ok := cache.get(mlReq.Query)
	if !ok {
		exec := h.sctx.(sqlexec.RestrictedSQLExecutor)
		stmt, err := exec.ParseWithParamsInternal(context.Background(), mlReq.Query)
		if err != nil {
			return nil, fmt.Errorf("invalid query=%v, err=%v", mlReq.Query, err)
		}
		rows, _, err := exec.ExecRestrictedStmt(context.Background(), stmt)
		if err != nil {
			return nil, err
		}

		if len(mlReq.DataPartitionMap) > 1 {
			// TODO: select a part of data randomly to train
		}

		xT, yT, err = convert4Iris(rows)
		if err != nil {
			return nil, err
		}
	}

	decodeDuf := bytes.NewBuffer(mlReq.ModelData)
	dec := gob.NewDecoder(decodeDuf)
	var thetaT *tensor.Dense
	err := dec.Decode(&thetaT)
	if err != nil {
		return nil, err
	}

	g := gorgonia.NewGraph()
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	y := gorgonia.NodeFromAny(g, yT, gorgonia.WithName("y"))
	theta := gorgonia.NewVector(
		g,
		gorgonia.Float64,
		gorgonia.WithName("theta"),
		gorgonia.WithShape(xT.Shape()[1]),
		gorgonia.WithValue(thetaT))

	pred := must(gorgonia.Mul(x, theta))
	// Saving the value for later use
	var predicted gorgonia.Value
	gorgonia.Read(pred, &predicted)
	squaredError := must(gorgonia.Square(must(gorgonia.Sub(pred, y))))
	cost := must(gorgonia.Mean(squaredError))
	if _, err := gorgonia.Grad(cost, theta); err != nil {
		logMaster("Failed to backpropagate: %v", err)
		os.Exit(0)
	}

	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.001))
	model := []gorgonia.ValueGrad{theta}

	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(theta))
	defer machine.Close()

	if err = machine.RunAll(); err != nil {
		fmt.Printf("Error during iteration: %v: %v\n", mlReq.Iter, err)
		return nil, err
	}

	if err = solver.Step(model); err != nil {
		fmt.Println(">>>>>> ", err)
		os.Exit(0)
	}
	machine.Reset() // Reset is necessary in a loop like this

	fmt.Printf("theta: %2.2f  Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
		theta.Value(),
		mlReq.Iter,
		cost.Value(),
		accuracy(predicted.Data().([]float64), y.Value().Data().([]float64)))

	var encodeBuf bytes.Buffer
	enc := gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(theta.Value()); err != nil {
		return nil, err
	}
	return encodeBuf.Bytes(), nil
}
