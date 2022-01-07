package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"gorgonia.org/tensor"
	"strings"

	"github.com/pingcap/tidb/distsql"
	"github.com/pingcap/tidb/infoschema"
	"github.com/pingcap/tidb/kv"
	plannercore "github.com/pingcap/tidb/planner/core"
	"github.com/pingcap/tidb/util/chunk"
	"github.com/pingcap/tidb/util/logutil"
	"github.com/pingcap/tidb/util/sqlexec"
	"gorgonia.org/gorgonia"
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

	// start to training this model
	modelData, err := ml.train(ctx, model, paraData)
	if err != nil {
		return err
	}

	_, err = exec.ExecuteInternal(ctx, "update mysql.ml_models set model_data = %? where name = %?", modelData, ml.v.Model)
	return err
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
	logutil.BgLogger().Info(fmt.Sprintf("numFeatures = %v, numClasses = %v, hiddenUnits = %v, batchSize = %v, learningRate = %v", params.numFeatures, params.numClasses, params.hiddenUnits, params.batchSize, params.learningRate))

	g, _, _, learnables, err := constructModel(params)
	if err != nil {
		return nil, err
	}

	// compile graph and construct machine
	_, _, err = gorgonia.Compile(g)
	if err != nil {
		return nil, err
	}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(params.learningRate))

	// TODO: init model data: yifan, lanhai
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

		gradValues := convertGradValues(learnables, avgGrads)
		if err := solver.Step(gradValues); err != nil {
			return nil, err
		}

		// TODO: update the model data: yifan, lanhai
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
	for j := 0; j < len(slaverGrads[0]); j++ {
		grad := slaverGrads[0][j].(*tensor.Dense)
		gradValue := grad.Data().([]float64)
		nGradValue := make([]float64, 0, len(gradValue))
		nGradValue = append(nGradValue, gradValue...)
		for i := 1; i < len(slaverGrads); i++ {
			grad = slaverGrads[i][j].(*tensor.Dense)
			gradValue = grad.Data().([]float64)
			for k := 0; k < len(nGradValue); k++ {
				nGradValue[k] = nGradValue[k] + gradValue[k]
			}
		}
		for k := 0; k < len(nGradValue); k++ {
			nGradValue[k] = nGradValue[k] / float64(len(slaverGrads))
		}
		nGrad := tensor.NewDense(tensor.Float64, []int{len(nGradValue),1}, tensor.WithBacking(nGradValue))
		values = append(values, nGrad)
	}
	return values, nil
}

func convertGradValues(learnables []*gorgonia.Node, values []gorgonia.Value) []gorgonia.ValueGrad {
	for i := 0; i < len(learnables); i++ {
		learnables[i].SetGrad(values[i])
	}
	return gorgonia.NodesToValueGrads(learnables)
}

func (ml *MLTrainModelExecutor) constructMLReq(iter int, dataPartitionMap map[string]int, model, modelParameters string, modelData []byte) (*kv.Request, error) {
	var builder distsql.RequestBuilder
	mlReq := &MLModelReq{iter, dataPartitionMap, model, modelParameters, modelData, ml.v.Query}
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
	ModelType        string
	Parameters       string // json format of map[string]string
	ModelData        []byte // encoding from []gorgonia.Value
	Query            string
}
