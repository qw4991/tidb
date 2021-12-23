package executor

import plannercore "github.com/pingcap/tidb/planner/core"

type MLCreateModelExecutor struct {
	baseExecutor

	v *plannercore.MLCreateModel
}

type MLTrainModelExecutor struct {
	baseExecutor

	v *plannercore.MLTrainModel
}
