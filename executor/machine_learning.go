package executor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/pingcap/tidb/distsql"
	"github.com/pingcap/tidb/infoschema"
	"github.com/pingcap/tidb/kv"
	"strings"

	plannercore "github.com/pingcap/tidb/planner/core"
	"github.com/pingcap/tidb/util/chunk"
	"github.com/pingcap/tidb/util/sqlexec"
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

	// TODO: save this model
	_, err = exec.ExecuteInternal(ctx, "update mysql.ml_models set model_data = %? where name = %?", modelData, ml.v.Model)
	return err
}

func (ml *MLTrainModelExecutor) train(ctx context.Context, modelType, parameters string) ([]byte, error) {
	// data partition
	dataPartitionMap, err := ml.constructDataPartitionMap()
	if err != nil {
		return nil, err
	}

	// TODO: init model data
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

		// TODO: update the model data
	}

	return modelData, nil
}

func (ml *MLTrainModelExecutor) constructMLReq(iter int, dataPartitionMap map[string]int, modelType, modelParameters string, modelData []byte) (*kv.Request, error) {
	var builder distsql.RequestBuilder
	mlReq := &MLModelReq{iter, dataPartitionMap, modelType, modelParameters, modelData}
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
	Parameters       string
	ModelData        []byte
}

func HandleSlaverTrainingReq(req []byte) ([]byte, error) {
	var mlReq MLModelReq
	if err := json.Unmarshal(req, &mlReq); err != nil {
		return nil, err
	}
	fmt.Println(">>>>>>>>>>> receive req >> ", mlReq)
	return nil, nil
}
