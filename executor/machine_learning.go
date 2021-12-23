package executor

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	plannercore "github.com/pingcap/tidb/planner/core"
	"github.com/pingcap/tidb/util/chunk"
	"github.com/pingcap/tidb/util/sqlexec"
)

type MLCreateModelExecutor struct {
	baseExecutor

	v         *plannercore.MLCreateModel
	modelType string
	paraMap   map[string]string
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
	sql := fmt.Sprintf("insert into mysql.ml_models values ('%v', '%v', '%v', NULL)", ml.v.Model, ml.modelType, string(paras))
	exec := ml.ctx.(sqlexec.SQLExecutor)
	_, err = exec.Execute(ctx, sql)
	return err
}

type MLTrainModelExecutor struct {
	baseExecutor

	v *plannercore.MLTrainModel
}

func (ml *MLTrainModelExecutor) Next(ctx context.Context, req *chunk.Chunk) error {
	sql := fmt.Sprintf("select type, parameters from mysql.ml_models where name='%v'", ml.v.Model)
	exec := ml.ctx.(sqlexec.SQLExecutor)
	rs, err := exec.Execute(ctx, sql)
	if err != nil {
		return err
	}
	if len(rs) == 0 {
		return errors.New(fmt.Sprintf("model %v not found", ml.v.Model))
	}
	sRows, err := resultSetToStringSlice(context.Background(), rs[0])
	modelType, paraData := sRows[0][0], sRows[0][1]
	fmt.Println(">>>> ", modelType, paraData)
	return nil
}
