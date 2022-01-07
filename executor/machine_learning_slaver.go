package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sync"

	"github.com/pingcap/errors"
	"github.com/pingcap/tidb/config"
	"github.com/pingcap/tidb/sessionctx"
	"github.com/pingcap/tidb/util"
	"github.com/pingcap/tidb/util/kvcache"
	"github.com/pingcap/tidb/util/sqlexec"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// train model name with query
func (h *CoprocessorDAGHandler) HandleSlaverTrainingReq(req []byte) ([]byte, error) {
	var mlReq MLModelReq
	if err := json.Unmarshal(req, &mlReq); err != nil {
		return nil, errors.Trace(err)
	}

	if mlReq.CaseType == "iris" {
		return h.HandleSlaverTraining4Iris(mlReq)
	}

	self := fmt.Sprintf("%v:%v %v", util.GetLocalIP(), config.GetGlobalConfig().Port, config.GetGlobalConfig().Store)
	//logSlaver(self, "model data length %v", len(mlReq.ModelData))

	// parse parameters
	var paramMap map[string]string
	if err := json.Unmarshal([]byte(mlReq.Parameters), &paramMap); err != nil {
		return nil, errors.New("encounter error when decoding parameters")
	}
	params, err := parseModelParams(mlReq.ModelType, paramMap)
	if err != nil {
		return nil, errors.Trace(err)
	}

	var (
		g          *gorgonia.ExprGraph
		x, y       *gorgonia.Node
		learnables []*gorgonia.Node
		loss       *gorgonia.Node
	)

	if mlReq.CaseType == "iris" {
		g, x, y, learnables, loss, err = constructModel4Iris()
	}
	g, x, y, learnables, loss, err = constructModel2(params)
	if err != nil {
		return nil, errors.Trace(err)
	}

	// TODO: loss function and optimizer/solver can also be added in params
	//logSlaver(self, "numFeatures = %v, numClasses = %v, hiddenUnits = %v, batchSize = %v, learningRate = %v",
	//	params.numFeatures, params.numClasses, params.hiddenUnits, params.batchSize, params.learningRate)

	// compile graph and construct machine
	prog, locMap, err := gorgonia.Compile(g)
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(learnables...))

	// decode weights from mlReq.ModelData and assign them to learnables
	//decodeDuf := bytes.NewBuffer(mlReq.ModelData)
	//decoder := gob.NewDecoder(decodeDuf)
	//var weights []gorgonia.Value
	//if err = decoder.Decode(&weights); err != nil {
	//	return nil, errors.Trace(err)
	//}
	//for i, weight := range weights {
	//	if err = gorgonia.Let(learnables[i], weight); err != nil {
	//		return nil, errors.Trace(err)
	//	}
	//}

	// TODO: read data: yuanjia, cache
	xVal, yVal, err := readMLData(h.sctx, params.batchSize, mlReq.Query, mlReq.CaseType)
	if err != nil {
		return nil, errors.Trace(err)
	}

	// TODO: convert rows to xVal, yVal
	if err = gorgonia.Let(x, xVal); err != nil {
		return nil, errors.Trace(err)
	}
	if err = gorgonia.Let(y, yVal); err != nil {
		return nil, errors.Trace(err)
	}

	// TODO: for debug <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(params.learningRate))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(params.batchSize)))
	for i := 0; i < 100; i++ {
		if err = vm.RunAll(); err != nil {
			return nil, errors.Trace(err)
		}

		logSlaver(self, "loss = %v", loss.Value().Data())

		valueGrads := gorgonia.NodesToValueGrads(learnables)
		if err := solver.Step(valueGrads); err != nil {
			panic(err)
		}
		vm.Reset()
	}
	os.Exit(0)
	// TODO: for debug <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

	// TODO: train the model with mlReq and return gradients: yifan, lanhai
	if err = vm.RunAll(); err != nil {
		return nil, errors.Trace(err)
	}

	logSlaver(self, "loss = %v", loss.Value().Data())

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

type mlDataKey string

func (k mlDataKey) Hash() []byte {
	return []byte(k)
}

type mlDataVal struct {
	x *tensor.Dense
	y *tensor.Dense
}

type mlDataCache struct {
	cache *kvcache.SimpleLRUCache
	mu    sync.RWMutex
}

func newMLDataCache(cap uint) *mlDataCache {
	return &mlDataCache{
		cache: kvcache.NewSimpleLRUCache(cap, 0.1, math.MaxUint64),
	}
}

func (c *mlDataCache) get(query string) (*tensor.Dense, *tensor.Dense, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	v, ok := c.cache.Get(mlDataKey(query))
	if !ok {
		return nil, nil, false
	}
	mlVal := v.(*mlDataVal)
	return mlVal.x, mlVal.y, true
}

func (c *mlDataCache) put(query string, x, y *tensor.Dense) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache.Put(mlDataKey(query), &mlDataVal{x, y})
}

var cache *mlDataCache

func init() {
	const MB = 1 << 20
	cache = newMLDataCache(MB * 16)
}

func readMLData(sctx sessionctx.Context, batchSize int, query string, caseType string) (x *tensor.Dense, y *tensor.Dense, err error) {
	// read from the cache
	if x, y, ok := cache.get(query); ok {
		return x, y, nil
	}
	defer func() {
		if err == nil {
			cache.put(query, x, y)
		}
	}()

	// read from TiKV
	exec := sctx.(sqlexec.RestrictedSQLExecutor)
	stmt, err := exec.ParseWithParamsInternal(context.Background(), query)
	if err != nil {
		return nil, nil, errors.Errorf("invalid query=%v, err=%v", query, err)
	}
	rows, fields, err := exec.ExecRestrictedStmt(context.Background(), stmt)
	if err != nil {
		return nil, nil, errors.Trace(err)
	}
	// TODO: only support 2 columns (img, label) for minist-training data now
	//if len(fields) != 2 && fields[0].ColumnAsName.L != "img" && fields[1].ColumnAsName.L != "label" {
	//	return nil, nil, errors.Errorf("unsupported training query %v", query)
	//}

	if caseType == "iris" {
		panic("???")
	}

	n := len(rows)
	ac := len(fields)
	vl := 28 * 28
	yl := 10
	if ac > 2 {
		vl = 4
		yl = 3
	}
	xVal, yVal := make([]float64, 0, n*vl), make([]float64, 0, n*yl)
	for i, r := range rows {
		if i == batchSize {
			break
		}
		if ac == 2 {
			vx := r.GetBytes(0)
			for _, v := range vx {
				xVal = append(xVal, float64(v))
			}
		} else {
			for j := 0; j < ac-1; j++ {
				xVal = append(xVal, r.GetFloat64(j))
			}
		}
		vy := r.GetFloat64(ac - 1)
		label := int(vy)
		for j := 0; j < yl; j++ {
			if j == label {
				yVal = append(yVal, 0.9)
			} else {
				yVal = append(yVal, 0.1)
			}
		}
	}

	y = tensor.New(tensor.WithShape(batchSize, yl), tensor.WithBacking(yVal))
	// TODO: 28*28
	x = tensor.New(tensor.WithShape(batchSize, vl), tensor.WithBacking(xVal))
	return x, y, nil
}
