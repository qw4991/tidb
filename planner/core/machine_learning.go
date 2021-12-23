package core

import (
	"github.com/pingcap/tidb/parser/ast"
)

type LogicalCreateModel struct {
	baseSchemaProducer

	Model      string
	Parameters []string
}

func NewLogicalCreateModel(stmt *ast.CreateModelStmt) *LogicalCreateModel {
	return &LogicalCreateModel{
		Model:      stmt.Name.O,
		Parameters: stmt.Parameters,
	}
}

type LogicalTrainModel struct {
	baseSchemaProducer

	Model string
	Query string
}

func NewLogicalTrainModel(stmt *ast.TrainModelStmt) *LogicalTrainModel {
	return &LogicalTrainModel{
		Model: stmt.Name.O,
		Query: stmt.Query,
	}
}
