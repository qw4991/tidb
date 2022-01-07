package executor

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/pingcap/tidb/config"
	"github.com/pingcap/tidb/util"
	"github.com/pingcap/tidb/util/logutil"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

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
	learnables := make([]*gorgonia.Node, 0, len(hiddenUnits)+1)
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
	xVal, yVal := readMLData(mlReq.Query)

	// TODO: convert rows to xVal, yVal
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

func readMLData(query string) (*tensor.Dense, *tensor.Dense) {
	// TODO: yuanjia

	return nil, nil
}
