package executor

import (
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"github.com/pingcap/errors"
	"github.com/pingcap/tidb/config"
	"github.com/pingcap/tidb/sessionctx"
	"github.com/pingcap/tidb/util"
	"github.com/pingcap/tidb/util/kvcache"
	"github.com/pingcap/tidb/util/logutil"
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

	// TODO: init the model: yifan, lanhai

	// TODO: maybe model type can also be stored in params map.
	// parse parameters
	var paramMap map[string]string
	if err := json.Unmarshal([]byte(mlReq.Parameters), &paramMap); err != nil {
		return nil, errors.New("encounter error when decoding parameters")
	}
	params, err := parseModelParams(mlReq.ModelType, paramMap)
	if err != nil {
		return nil, errors.Trace(err)
	}
	// TODO: loss function and optimizer/solver can also be added in params
	logutil.BgLogger().Info(fmt.Sprintf("numFeatures = %v, numClasses = %v, hiddenUnits = %v, batchSize = %v, learningRate = %v", params.numFeatures, params.numClasses, params.hiddenUnits, params.batchSize, params.learningRate))

	g, x, y, learnables, err := constructModel(params)
	if err != nil {
		return nil, errors.Trace(err)
	}

	// compile graph and construct machine
	prog, locMap, err := gorgonia.Compile(g)
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(learnables...))

	// decode weights from mlReq.ModelData and assign them to learnables
	decodeDuf := bytes.NewBuffer(mlReq.ModelData)
	decoder := gob.NewDecoder(decodeDuf)
	var weights []gorgonia.Value
	if err = decoder.Decode(&weights); err != nil {
		return nil, errors.Trace(err)
	}
	for i, weight := range weights {
		if err = gorgonia.Let(learnables[i], weight); err != nil {
			return nil, errors.Trace(err)
		}
	}

	self := fmt.Sprintf("%v:%v %v", util.GetLocalIP(), config.GetGlobalConfig().Port, config.GetGlobalConfig().Store)
	fmt.Println(">>>>>>>>>>> receive req >> ", self, len(mlReq.ModelData))

	// TODO: read data: yuanjia, cache
	xVal, yVal, err := readMLData(h.sctx, params.batchSize, mlReq.Query)
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

	// TODO: train the model with mlReq and return gradients: yifan, lanhai
	if err = vm.RunAll(); err != nil {
		return nil, errors.Trace(err)
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

func readMLData(sctx sessionctx.Context, batchSize int, query string) (x *tensor.Dense, y *tensor.Dense, err error) {
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
	if len(fields) != 2 && fields[0].ColumnAsName.L != "img" && fields[1].ColumnAsName.L != "label" {
		return nil, nil, errors.Errorf("unsupported training query %v", query)
	}

	n := len(rows)
	xVal, yVal := make([]float64, 0, n*28*28), make([]float64, 0, n*10)
	for i, r := range rows {
		if i == batchSize {
			break
		}
		vx := r.GetBytes(0)
		vy := r.GetFloat64(1)
		for _, v := range vx {
			xVal = append(xVal, float64(v))
		}
		label := int(vy)
		for j := 0; j < 10; j++ {
			if j == label {
				yVal = append(yVal, 0.9)
			} else {
				yVal = append(yVal, 0.1)
			}
		}
	}

	y = tensor.New(tensor.WithShape(batchSize, 10), tensor.WithBacking(yVal))
	// TODO: 28*28
	x = tensor.New(tensor.WithShape(batchSize, 28*28), tensor.WithBacking(xVal))
	return x, y, nil
}
