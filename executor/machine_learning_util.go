package executor

import (
	"fmt"
	"github.com/pingcap/errors"
	"gorgonia.org/gorgonia"
	"strconv"
	"strings"
)

type modelType int

const (
	dnnClassifier modelType = iota
	convNet
)

type modelParams struct {
	model        modelType
	numFeatures  int
	numClasses   int
	hiddenUnits  []int
	batchSize    int
	learningRate float64
}

func string2IntSlice(str string) ([]int, error) {
	// [1, 2, 3, 4]
	str = strings.TrimSpace(str)
	str = str[1 : len(str)-1]
	strSlice := strings.Split(str, ",")
	intSlice := make([]int, 0, len(strSlice))
	for _, s := range strSlice {
		i, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			return nil, errors.Errorf("invalid intStr=%v", s)
		}
		intSlice = append(intSlice, i)
	}
	return intSlice, nil
}

func parseModelParams(model string, paramMap map[string]string) (modelParams, error) {
	res := modelParams{}
	var err error
	if model != "DNNClassifier" {
		return res, errors.New("unsupported model")
	}
	res.model = dnnClassifier
	strNumFeatures, ok := paramMap["n_features"]
	if !ok {
		return res, errors.New("n_features it not specified")
	}
	res.numFeatures, err = strconv.Atoi(strNumFeatures)
	if err != nil {
		return res, errors.New("n_features must be an integer")
	}
	strNumClasses, ok := paramMap["n_classes"]
	if !ok {
		return res, errors.New("n_classes is not specified")
	}
	res.numClasses, err = strconv.Atoi(strNumClasses)
	if err != nil {
		return res, errors.New("n_classes must be an integer")
	}
	strHiddenUnits, ok := paramMap["hidden_units"]
	if !ok {
		return res, errors.New("hidden_units is not specified")
	}
	res.hiddenUnits, err = string2IntSlice(strHiddenUnits)
	if err != nil {
		return res, errors.Errorf("invalid hidden_units=%v, it must be an array of integers like [2,3,5], err=%v", strHiddenUnits, err)
	}
	strBatchSize, ok := paramMap["batch_size"]
	if !ok {
		return res, errors.New("batch_size is not specified")
	}
	res.batchSize, err = strconv.Atoi(strBatchSize)
	if err != nil {
		return res, errors.New("batch_size must be an integer")
	}
	strLearningRate, ok := paramMap["learning_rate"]
	if !ok {
		return res, errors.New("learning_rate is not specified")
	}
	res.learningRate, err = strconv.ParseFloat(strLearningRate, 64)
	if err != nil {
		return res, errors.New("learning_size must be a float")
	}
	return res, nil
}

func constructModel(params modelParams) (g *gorgonia.ExprGraph, x, y *gorgonia.Node, learnables []*gorgonia.Node, err error) {
	// construct the computation graph
	g = gorgonia.NewGraph()
	x = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(params.batchSize, params.numFeatures), gorgonia.WithName("x"))
	y = gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(params.batchSize, params.numClasses), gorgonia.WithName("y"))
	learnables = make([]*gorgonia.Node, 0, len(params.hiddenUnits)+1)
	weightNum := 0
	current := x
	currentLen := x.Shape()[1]
	for _, hiddenLen := range params.hiddenUnits {
		w := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(currentLen, hiddenLen), gorgonia.WithName(fmt.Sprintf("w%v", weightNum)), gorgonia.WithInit(gorgonia.Gaussian(0, 0.1)))
		weightNum++
		learnables = append(learnables, w)
		currentLen = hiddenLen
		current, err = gorgonia.Mul(current, w)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		current, err = gorgonia.Rectify(current)
		if err != nil {
			return nil, nil, nil, nil, err
		}
		current, err = gorgonia.Dropout(current, 0.5)
		if err != nil {
			return nil, nil, nil, nil, err
		}
	}
	w := gorgonia.NewVector(g, gorgonia.Float64, gorgonia.WithShape(currentLen, params.numClasses), gorgonia.WithName(fmt.Sprintf("w%v", weightNum)), gorgonia.WithInit(gorgonia.Gaussian(0, 0.1)))
	weightNum++
	learnables = append(learnables, w)
	current, err = gorgonia.Mul(current, w)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	current, err = gorgonia.SoftMax(current)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// cross entropy loss
	current, err = gorgonia.Log(current)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	current, err = gorgonia.Neg(current)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	current, err = gorgonia.HadamardProd(current, y)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	loss, err := gorgonia.Mean(current)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	_, err = gorgonia.Grad(loss, learnables...)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return
}
