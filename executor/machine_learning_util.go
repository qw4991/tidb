package executor

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/pingcap/errors"
	"github.com/pingcap/tidb/util/logutil"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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

func constructModel4Iris() (g *gorgonia.ExprGraph, x, y *gorgonia.Node, learnables []*gorgonia.Node, loss *gorgonia.Node, err error) {
	return nil, nil, nil, nil, nil, nil
}

func constructModel(params modelParams) (g *gorgonia.ExprGraph, x, y *gorgonia.Node, learnables []*gorgonia.Node, loss *gorgonia.Node, err error) {
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
			return nil, nil, nil, nil, nil, errors.Trace(err)
		}
		current, err = gorgonia.Rectify(current)
		if err != nil {
			return nil, nil, nil, nil, nil, errors.Trace(err)
		}
		current, err = gorgonia.Dropout(current, 0.5)
		if err != nil {
			return nil, nil, nil, nil, nil, errors.Trace(err)
		}
	}
	w := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(currentLen, params.numClasses), gorgonia.WithName(fmt.Sprintf("w%v", weightNum)), gorgonia.WithInit(gorgonia.Gaussian(0, 0.1)))
	weightNum++
	learnables = append(learnables, w)
	current, err = gorgonia.Mul(current, w)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	current, err = gorgonia.SoftMax(current)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}

	// cross entropy loss
	current, err = gorgonia.Log(current)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	current, err = gorgonia.Neg(current)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	current, err = gorgonia.HadamardProd(current, y)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	loss, err = gorgonia.Mean(current)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	_, err = gorgonia.Grad(loss, learnables...)
	if err != nil {
		return nil, nil, nil, nil, nil, errors.Trace(err)
	}
	return
}

func logMaster(format string, vals ...interface{}) {
	logutil.BgLogger().Info(fmt.Sprintf("[ML_Master] "+format, vals...))
}

func logSlaver(addr, format string, vals ...interface{}) {
	logutil.BgLogger().Info(fmt.Sprintf("[ML_Slaver:"+addr+"] "+format, vals...))
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
		return errors.Wrap(err, "Layer 0 Convolution failed")
	}
	if a0, err = gorgonia.Rectify(c0); err != nil {
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	if p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	if l0, err = gorgonia.Dropout(p0, m.d0); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout")
	}

	// Layer 1
	if c1, err = gorgonia.Conv2d(l0, m.w1, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if a1, err = gorgonia.Rectify(c1); err != nil {
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if l1, err = gorgonia.Dropout(p1, m.d1); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}

	// Layer 2
	if c2, err = gorgonia.Conv2d(l1, m.w2, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if a2, err = gorgonia.Rectify(c2); err != nil {
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	if p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 2 Maxpooling failed")
	}

	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	if r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 2")
	}
	if l2, err = gorgonia.Dropout(r2, m.d2); err != nil {
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}

	// Layer 3
	if fc, err = gorgonia.Mul(l2, m.w3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a3, err = gorgonia.Rectify(fc); err != nil {
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if l3, err = gorgonia.Dropout(a3, m.d3); err != nil {
		return errors.Wrapf(err, "Unable to apply a dropout on layer 3")
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

func constructModel2(params modelParams) (g *gorgonia.ExprGraph, x, y *gorgonia.Node, learnables []*gorgonia.Node, loss *gorgonia.Node, err error) {
	g = gorgonia.NewGraph()
	x = gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(params.batchSize, 1, 28, 28), gorgonia.WithName("x"))
	y = gorgonia.NewMatrix(g, dt, gorgonia.WithShape(params.batchSize, 10), gorgonia.WithName("y"))
	m := newConvNet(g)
	if err = m.fwd(x); err != nil {
		return nil, nil, nil, nil, nil, err
	}

	losses := gorgonia.Must(gorgonia.Log(gorgonia.Must(gorgonia.HadamardProd(m.out, y))))
	loss = gorgonia.Must(gorgonia.Mean(losses))
	loss = gorgonia.Must(gorgonia.Neg(loss))

	if _, err = gorgonia.Grad(loss, m.learnables()...); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	return
}
