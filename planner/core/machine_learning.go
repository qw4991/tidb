package core

import (
	"github.com/pingcap/tidb/parser/ast"
)

// TODO: support show models

type MLCreateModel struct {
	baseSchemaProducer

	Model      string
	Parameters []string
}

func NewMLCreateModel(stmt *ast.CreateModelStmt) *MLCreateModel {
	return &MLCreateModel{
		Model:      stmt.Name.O,
		Parameters: stmt.Parameters,
	}
}

type MLTrainModel struct {
	baseSchemaProducer

	Model string
	Query string
}

func NewMLTrainModel(stmt *ast.TrainModelStmt) *MLTrainModel {
	return &MLTrainModel{
		Model: stmt.Name.O,
		Query: stmt.Query,
	}
}

type MLSlaverTrainModel struct {
	physicalSchemaProducer

	ModelType string
	ParamData string
}
