package executor

import (
	"context"
	"errors"
	plannercore "github.com/pingcap/tidb/planner/core"
	"strings"
)

type MLCreateModelExecutor struct {
	baseExecutor

	v *plannercore.MLCreateModel
}

func (ml *MLCreateModelExecutor) Open(ctx context.Context) error {
	parameters := ml.v.Parameters
	pm := make(map[string]string)
	for i := 0; i < len(parameters); i += 2 {
		pm[strings.ToLower(parameters[i])] = pm[parameters[i+1]]
	}
	if _, ok := pm["type"]; !ok {
		return errors.New("no type parameter")
	}



	return nil
}

type MLTrainModelExecutor struct {
	baseExecutor

	v *plannercore.MLTrainModel
}
