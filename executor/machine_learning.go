package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/pingcap/tidb/distsql"
	"github.com/pingcap/tidb/infoschema"
	"github.com/pingcap/tidb/kv"
	plannercore "github.com/pingcap/tidb/planner/core"
	"github.com/pingcap/tidb/util/chunk"
	"github.com/pingcap/tidb/util/sqlexec"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"math"
	"os"
	"strings"
)

type MLCreateModelExecutor struct {
	baseExecutor

	v       *plannercore.MLCreateModel
	paraMap map[string]string
}

func (ml *MLCreateModelExecutor) Open(ctx context.Context) error {
	ml.paraMap = make(map[string]string)
	for i := 0; i < len(ml.v.Parameters); i += 2 {
		ml.paraMap[strings.ToLower(ml.v.Parameters[i])] = ml.v.Parameters[i+1]
	}
	if _, ok := ml.paraMap["type"]; !ok {
		return errors.New("no type parameter")
	}
	// TODO: check whether other parameters are valid
	return nil
}

func (ml *MLCreateModelExecutor) Next(ctx context.Context, req *chunk.Chunk) error {
	paras, err := json.Marshal(ml.paraMap)
	if err != nil {
		return err
	}
	sql := fmt.Sprintf("insert into mysql.ml_models values ('%v', '%v', '%v', NULL)", ml.v.Model, ml.paraMap["type"], string(paras))
	exec := ml.ctx.(sqlexec.SQLExecutor)
	_, err = exec.Execute(ctx, sql)
	return err
}

type MLTrainModelExecutor struct {
	baseExecutor

	v *plannercore.MLTrainModel
}

func (ml *MLTrainModelExecutor) Next(ctx context.Context, req *chunk.Chunk) error {
	// read information about this model
	exec := ml.ctx.(sqlexec.SQLExecutor)
	rs, err := exec.ExecuteInternal(ctx, "select type, parameters from mysql.ml_models where name=%?", ml.v.Model)
	if err != nil {
		return err
	}
	sRows, err := resultSetToStringSlice(context.Background(), rs)
	if err != nil {
		return err
	}
	if len(sRows) == 0 {
		return errors.New(fmt.Sprintf("model %v not found", ml.v.Model))
	}
	model, paraData := sRows[0][0], sRows[0][1]

	var modelData []byte
	if strings.Contains(strings.ToLower(ml.v.Model), "iris") {
		//modelData, err = ml.train4Iris(ctx)
		modelData, err = ml.train4Iris2(ctx)
	} else {
		// start to training this model
		modelData, err = ml.train(ctx, model, paraData)
	}
	if err != nil {
		return err
	}

	_, err = exec.ExecuteInternal(ctx, "update mysql.ml_models set model_data = %? where name = %?", modelData, ml.v.Model)
	return err
}

func (ml *MLTrainModelExecutor) train4Iris2(ctx context.Context) ([]byte, error) {
	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(150, 4))
	y := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(150))
	theta := gorgonia.NewVector(
		g,
		gorgonia.Float64,
		gorgonia.WithName("theta"),
		gorgonia.WithShape(4),
		gorgonia.WithInit(gorgonia.Uniform(0, 1)))

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

	var encodeBuf bytes.Buffer
	enc := gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(theta.Value()); err != nil {
		return nil, err
	}
	modelData := encodeBuf.Bytes()

	iter := 10000
	for i := 0; i < iter; i++ {
		req, err := ml.constructMLReq(iter, nil, "DNNClassifier", "{}", modelData)
		if err != nil {
			return nil, err
		}
		resp := ml.ctx.GetClient().Send(ctx, req, ml.ctx.GetSessionVars().KVVars, ml.ctx.GetSessionVars().StmtCtx.MemTracker, false, nil)
		defer resp.Close()

		var slaverGrads []gorgonia.Value
		for {
			data, err := resp.Next(ctx)
			if err != nil {
				return nil, err
			}
			if data == nil { // no more data
				break
			}
			decodeDuf := bytes.NewBuffer(data.GetData())
			decoder := gob.NewDecoder(decodeDuf)
			var grads *tensor.Dense
			if err = decoder.Decode(&grads); err != nil {
				return nil, err
			}

			slaverGrads = append(slaverGrads, grads)
		}

		var avgGrad gorgonia.Value
		if len(slaverGrads) == 1 {
			avgGrad = slaverGrads[0]
		} else {
			panic("TODO")
		}

		if err := theta.SetGrad(avgGrad); err != nil {
			return nil, err
		}

		if err := solver.Step(model); err != nil {
			return nil, err
		}

		encodeBuf.Reset()
		enc := gob.NewEncoder(&encodeBuf)
		if err := enc.Encode(theta.Value()); err != nil {
			return nil, err
		}
		modelData = encodeBuf.Bytes()
	}

	enc = gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(theta.Value()); err != nil {
		return nil, err
	}
	return encodeBuf.Bytes(), nil
}

func (ml *MLTrainModelExecutor) train4Iris(ctx context.Context) ([]byte, error) {
	exec := ml.ctx.(sqlexec.RestrictedSQLExecutor)
	stmt, err := exec.ParseWithParamsInternal(context.Background(), ml.v.Query)
	if err != nil {
		return nil, fmt.Errorf("invalid query=%v, err=%v", ml.v.Query, err)
	}
	rows, _, err := exec.ExecRestrictedStmt(context.Background(), stmt)
	if err != nil {
		return nil, err
	}

	xT, yT, err := convert4Iris(rows)
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
		gorgonia.WithInit(gorgonia.Uniform(0, 1)))
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

	iter := 10000
	for i := 0; i < iter; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
		}

		if err = solver.Step(model); err != nil {
			fmt.Println(">>>>>> ", err)
			os.Exit(0)
		}
		machine.Reset() // Reset is necessary in a loop like this

		fmt.Printf("theta: %2.2f  Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
			theta.Value(),
			i,
			cost.Value(),
			accuracy(predicted.Data().([]float64), y.Value().Data().([]float64)))
	}

	var encodeBuf bytes.Buffer
	enc := gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(theta.Value()); err != nil {
		return nil, err
	}
	return encodeBuf.Bytes(), nil
}

func must(n *gorgonia.Node, err error) *gorgonia.Node {
	if err != nil {
		panic(err)
	}
	return n
}

func accuracy(prediction, y []float64) float64 {
	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}
	return ok / float64(len(y))
}

func convert4Iris(rows []chunk.Row) (x *tensor.Dense, y *tensor.Dense, err error) {
	n := len(rows)
	xs := make([]float64, 0, n*4)
	ys := make([]float64, 0, n)
	for _, r := range rows {
		x1, x2, x3, x4, y := r.GetFloat32(0), r.GetFloat32(1), r.GetFloat32(2), r.GetFloat32(3), r.GetInt64(4)
		xs = append(xs, float64(x1), float64(x2), float64(x3), float64(x4))
		ys = append(ys, float64(y))
	}

	x = tensor.New(tensor.WithShape(n, 4), tensor.WithBacking(xs))
	y = tensor.New(tensor.WithShape(n), tensor.WithBacking(ys))
	return
}

func (ml *MLTrainModelExecutor) train(ctx context.Context, model, parameters string) ([]byte, error) {
	// data partition
	dataPartitionMap, err := ml.constructDataPartitionMap()
	if err != nil {
		return nil, err
	}

	// Init the model according to parameters: yifan, lanhai
	// parse parameters
	var paramMap map[string]string
	if err := json.Unmarshal([]byte(parameters), &paramMap); err != nil {
		return nil, errors.New("encounter error when decoding parameters")
	}
	params, err := parseModelParams(model, paramMap)
	if err != nil {
		return nil, err
	}
	// TODO: loss function and optimizer/solver can also be added in params
	//logMaster("numFeatures = %v, numClasses = %v, hiddenUnits = %v, batchSize = %v, learningRate = %v", params.numFeatures, params.numClasses, params.hiddenUnits, params.batchSize, params.learningRate)

	g, _, _, learnables, _, err := constructModel(params)
	if err != nil {
		return nil, err
	}

	// compile graph and construct machine
	_, _, err = gorgonia.Compile(g)
	if err != nil {
		return nil, err
	}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(params.learningRate))

	var modelData []byte
	weights := make([]gorgonia.Value, 0, len(params.hiddenUnits)+1)
	for _, node := range learnables {
		weights = append(weights, node.Value())
	}
	var encodeBuf bytes.Buffer
	enc := gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(weights); err != nil {
		return nil, err
	}
	modelData = encodeBuf.Bytes()

	for iter := 0; iter < 10000; iter++ {
		req, err := ml.constructMLReq(iter, dataPartitionMap, model, parameters, modelData)
		if err != nil {
			return nil, err
		}
		resp := ml.ctx.GetClient().Send(ctx, req, ml.ctx.GetSessionVars().KVVars, ml.ctx.GetSessionVars().StmtCtx.MemTracker, false, nil)
		defer resp.Close()

		var slaverGrads [][]gorgonia.Value
		for {
			data, err := resp.Next(ctx)
			if err != nil {
				return nil, err
			}
			if data == nil { // no more data
				break
			}
			decodeDuf := bytes.NewBuffer(data.GetData())
			decoder := gob.NewDecoder(decodeDuf)
			var grads []gorgonia.Value
			if err = decoder.Decode(&grads); err != nil {
				return nil, err
			}
			slaverGrads = append(slaverGrads, grads)
		}

		avgGrads, err := calAvgGrads(slaverGrads)
		if err != nil {
			return nil, err
		}

		gradValues, err := convertGradValues(learnables, avgGrads)
		if err != nil {
			return nil, err
		}
		if err := solver.Step(gradValues); err != nil {
			return nil, err
		}

		weights = weights[:0]
		for _, node := range learnables {
			weights = append(weights, node.Value())
		}
		encodeBuf.Reset()
		enc := gob.NewEncoder(&encodeBuf)
		if err := enc.Encode(weights); err != nil {
			return nil, err
		}
		modelData = encodeBuf.Bytes()
	}

	return modelData, nil
}

func calAvgGrads(slaverGrads [][]gorgonia.Value) ([]gorgonia.Value, error) {
	values := make([]gorgonia.Value, 0, 1)
	var err error = nil
	for j := 0; j < len(slaverGrads[0]); j++ {
		grad := slaverGrads[0][j].(*tensor.Dense)
		cGrad := grad.Clone()
		for i := 1; i < len(slaverGrads); i++ {
			grad = slaverGrads[i][j].(*tensor.Dense)
			cGrad, err = tensor.Add(cGrad, grad.Clone())
			if err != nil {
				return nil, err
			}
		}
		cGrad, err = tensor.Div(cGrad, float64(len(slaverGrads)))
		if err != nil {
			return nil, err
		}
		values = append(values, cGrad.(*tensor.Dense))
	}
	return values, nil
}

func convertGradValues(learnables []*gorgonia.Node, values []gorgonia.Value) ([]gorgonia.ValueGrad, error) {
	for i := 0; i < len(learnables); i++ {
		if err := learnables[i].SetGrad(values[i]); err != nil {
			return nil, err
		}
	}
	return gorgonia.NodesToValueGrads(learnables), nil
}

func (ml *MLTrainModelExecutor) constructMLReq(iter int, dataPartitionMap map[string]int, model, modelParameters string, modelData []byte) (*kv.Request, error) {
	var builder distsql.RequestBuilder

	caseType := ""
	if strings.Contains(strings.ToLower(ml.v.Model), "iris") {
		caseType = "iris"
	}

	mlReq := &MLModelReq{iter, dataPartitionMap, caseType, model, modelParameters, modelData, ml.v.Query}
	reqData, err := json.Marshal(mlReq)
	if err != nil {
		return nil, err
	}

	builder.SetStoreType(kv.TiDBML)
	builder.Data = reqData
	req, err := builder.Build()
	if err != nil {
		return nil, err
	}
	req.Tp = kv.ReqTypeML
	return req, nil
}

func (ml *MLTrainModelExecutor) constructDataPartitionMap() (map[string]int, error) {
	serversInfo, err := infoschema.GetClusterServerInfo(ml.ctx)
	if err != nil {
		return nil, err
	}
	m := make(map[string]int)
	for i, s := range serversInfo {
		m[s.Address] = i
	}
	return m, nil
}

type MLModelReq struct {
	Iter             int
	DataPartitionMap map[string]int
	CaseType         string
	ModelType        string
	Parameters       string // json format of map[string]string
	ModelData        []byte // encoding from []gorgonia.Value
	Query            string
}
