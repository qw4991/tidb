package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"gorgonia.org/tensor"
	"strconv"
	"strings"

	"github.com/pingcap/tidb/config"
	"github.com/pingcap/tidb/distsql"
	"github.com/pingcap/tidb/infoschema"
	"github.com/pingcap/tidb/kv"
	plannercore "github.com/pingcap/tidb/planner/core"
	"github.com/pingcap/tidb/util"
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
	modelType, paraData := sRows[0][0], sRows[0][1]
	fmt.Println(">>>> ", modelType, paraData)

	// start to training this model
	modelData, err := ml.train(ctx, modelType, paraData)
	if err != nil {
		return nil
	}

	_, err = exec.ExecuteInternal(ctx, "update mysql.ml_models set model_data = %? where name = %?", modelData, ml.v.Model)
	return err
}

func (ml *MLTrainModelExecutor) train(ctx context.Context, modelType, parameters string) ([]byte, error) {
	// data partition
	dataPartitionMap, err := ml.constructDataPartitionMap()
	if err != nil {
		return nil, err
	}

	// TODO: init the model accroding to parameters: yifan, lanhai

	// TODO: init model data: yifan, lanhai
	var modelData []byte

	for iter := 0; iter < 10000; iter++ {
		req, err := ml.constructMLReq(iter, dataPartitionMap, modelType, parameters, modelData)
		if err != nil {
			return nil, err
		}
		resp := ml.ctx.GetClient().Send(ctx, req, ml.ctx.GetSessionVars().KVVars, ml.ctx.GetSessionVars().StmtCtx.MemTracker, false, nil)
		defer resp.Close()

		for {
			data, err := resp.Next(ctx)
			if err != nil {
				return nil, err
			}
			if data == nil { // no more data
				break
			}
		}

		// TODO: update the model data: yifan, lanhai
	}

	return modelData, nil
}

func (ml *MLTrainModelExecutor) constructMLReq(iter int, dataPartitionMap map[string]int, modelType, modelParameters string, modelData []byte) (*kv.Request, error) {
	var builder distsql.RequestBuilder
	mlReq := &MLModelReq{iter, dataPartitionMap, modelType, modelParameters, modelData, ""}
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

func string2IntSlice(str string) ([]int, error) {
	strSlice := strings.Split(strings.Trim(str, " "), ",")
	intSlice := make([]int, 0, len(strSlice))
	for _, s := range strSlice {
		i, err := strconv.Atoi(strings.Trim(s, " "))
		if err != nil {
			return nil, err
		}
		intSlice = append(intSlice, i)
	}
	return intSlice, nil
}

// train model name with query
func (h *CoprocessorDAGHandler) HandleSlaverTrainingReq(req []byte) ([]byte, error) {
	var mlReq MLModelReq
	if err := json.Unmarshal(req, &mlReq); err != nil {
		return nil, err
	}

	// TODO: init the model: yifan, lanhai

	// TODO: maybe model type can also be stored in params map.
	// parse parameters
	if mlReq.ModelType != "DNNClassifier" {
		return nil, errors.New("unsupported model")
	}
	var params map[string]string
	if err := json.Unmarshal([]byte(mlReq.Parameters), &params); err != nil {
		return nil, errors.New("encounter error when decoding parameters")
	}
	strNumFeatures, ok := params["n_features"]
	if !ok {
		return nil, errors.New("n_features it not specified")
	}
	numFeatures, err := strconv.Atoi(strNumFeatures)
	if err != nil {
		return nil, errors.New("n_features must be an integer")
	}
	strNumClasses, ok := params["n_classes"]
	if !ok {
		return nil, errors.New("n_classes is not specified")
	}
	numClasses, err := strconv.Atoi(strNumClasses)
	if err != nil {
		return nil, errors.New("n_classes must be an integer")
	}
	strHiddenUnits, ok := params["hidden_units"]
	if !ok {
		return nil, errors.New("hidden_units is not specified")
	}
	hiddenUnits, err := string2IntSlice(strHiddenUnits)
	if err != nil {
		return nil, errors.New("hidden_units must be an array of integers like [2,3,5]")
	}
	strBatchSize, ok := params["batch_size"]
	if !ok {
		return nil, errors.New("batch_size is not specified")
	}
	batchSize, err := strconv.Atoi(strBatchSize)
	if err != nil {
		return nil, errors.New("batch_size must be an integer")
	}
	strLearningRate, ok := params["learning_rate"]
	if !ok {
		return nil, errors.New("learning_rate is not specified")
	}
	learningRate, err := strconv.ParseFloat(strLearningRate, 64)
	if err != nil {
		return nil, errors.New("learning_size must be a float")
	}
	// TODO: loss function and optimizer/solver can also be added in params
	logutil.BgLogger().Info(fmt.Sprintf("numFeatures = %v, numClasses = %v, hiddenUnits = %v, batchSize = %v, learningRate = %v", numFeatures, numClasses, hiddenUnits, batchSize, learningRate))

	// construct the computation graph
	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(batchSize, numFeatures), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(batchSize, numClasses), gorgonia.WithName("y"))
	learnables := make([]*gorgonia.Node, 0, len(hiddenUnits) + 1)
	weightNum := 0
	current := x
	currentLen := x.Shape()[1]
	for _, hiddenLen := range hiddenUnits {
		w := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(currentLen, hiddenLen), gorgonia.WithName(fmt.Sprintf("w%v", weightNum)), gorgonia.WithInit(gorgonia.Gaussian(0, 0.1)))
		weightNum++
		learnables = append(learnables, w)
		currentLen = hiddenLen
		current, err = gorgonia.Mul(current, w)
		if err != nil {
			return nil, err
		}
		current, err = gorgonia.Rectify(current)
		if err != nil {
			return nil, err
		}
		current, err = gorgonia.Dropout(current, 0.5)
		if err != nil {
			return nil, err
		}
	}
	w := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(currentLen, numClasses), gorgonia.WithName(fmt.Sprintf("w%v", weightNum)), gorgonia.WithInit(gorgonia.Gaussian(0, 0.1)))
	weightNum++
	learnables = append(learnables, w)
	current, err = gorgonia.Mul(current, w)
	if err != nil {
		return nil, err
	}
	current, err = gorgonia.SoftMax(current)
	if err != nil {
		return nil, err
	}

	// cross entropy loss
	current, err = gorgonia.Log(current)
	if err != nil {
		return nil, err
	}
	current, err = gorgonia.Neg(current)
	if err != nil {
		return nil, err
	}
	current, err = gorgonia.HadamardProd(current, y)
	if err != nil {
		return nil, err
	}
	loss, err := gorgonia.Mean(current)
	if err != nil {
		return nil, err
	}
	_, err = gorgonia.Grad(loss, learnables...)
	if err != nil {
		return nil, err
	}

	// compile graph and construct machine
	prog, locMap, err := gorgonia.Compile(g)
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(learnables...))

	// decode weights from mlReq.ModelData and assign them to learnables
	decodeDuf := bytes.NewBuffer(mlReq.ModelData)
	decoder := gob.NewDecoder(decodeDuf)
	var weights []gorgonia.Value
	if err = decoder.Decode(&weights); err != nil {
		return nil, err
	}
	for i, weight := range weights {
		if err = gorgonia.Let(learnables[i], weight); err != nil {
			return nil, err
		}
	}

	self := fmt.Sprintf("%v:%v %v", util.GetLocalIP(), config.GetGlobalConfig().Port, config.GetGlobalConfig().Store)
	fmt.Println(">>>>>>>>>>> receive req >> ", self, mlReq)

	// TODO: read data: yuanjia, cache
	exec := h.sctx.(sqlexec.RestrictedSQLExecutor)
	stmt, err := exec.ParseWithParamsInternal(context.Background(), mlReq.Query)
	if err != nil {
		return nil, err
	}
	rows, fields, err := exec.ExecRestrictedStmt(context.Background(), stmt)
	if err != nil {
		return nil, err
	}
	// TODO: convert rows to xVal, yVal
	var xVal, yVal tensor.Tensor
	if err = gorgonia.Let(x, xVal); err != nil {
		return nil, err
	}
	if err = gorgonia.Let(y, yVal); err != nil {
		return nil, err
	}

	// TODO: train the model with mlReq and return gradients: yifan, lanhai
	if err = vm.RunAll(); err != nil {
		return nil, err
	}

	valueGrads := gorgonia.NodesToValueGrads(learnables)
	grads := make([]gorgonia.Value, 0, len(valueGrads))
	for _, valueGrad := range valueGrads {
		grad, err := valueGrad.Grad()
		if err != nil {
			return nil, err
		}
		grads = append(grads, grad)
	}

	var encodeBuf bytes.Buffer
	enc := gob.NewEncoder(&encodeBuf)
	if err := enc.Encode(grads); err != nil {
		return nil, err
	}

	return encodeBuf.Bytes(), nil
}
