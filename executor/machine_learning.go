package executor

import (
	"context"
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
}

func (ml *MLCreateModelExecutor) Open(ctx context.Context) error {
	hasType := false
	for i := 0; i < len(ml.v.Parameters); i += 2 {
		if strings.ToLower(ml.v.Parameters[i]) == "type" {
			hasType = true
			ml.modelType = ml.v.Parameters[i+1]
		}
	}
	if !hasType {
		return errors.New("no type parameter")
	}
	// TODO: check whether other parameters are valid
	return nil
}

func (ml *MLCreateModelExecutor) Next(ctx context.Context, req *chunk.Chunk) error {
	sql := fmt.Sprintf("insert into mysql.ml_models values ('%v', '%v', '%v', NULL)", ml.v.Model, ml.modelType, "test")
	exec := ml.ctx.(sqlexec.SQLExecutor)
	_, err := exec.Execute(ctx, sql)
	return err
}

type MLTrainModelExecutor struct {
	baseExecutor

	v *plannercore.MLTrainModel
}
