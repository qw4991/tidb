package executor

import (
	"context"
	"encoding/gob"
	"fmt"
	"github.com/pingcap/errors"
	"github.com/pingcap/tidb/util/sqlexec"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"math"
	"os"
)

// Image holds the pixel intensities of an image.
// 255 is foreground (black), 0 is background (white).
type RawImage []byte

// Label is a digit label in 0 to 9
type Label uint8

func (ml *MLTrainModelExecutor) train4Mnist(ctx context.Context) ([]byte, error) {
	var inputs, targets tensor.Tensor
	var err error

	fmt.Println(">>>>>>>>>>>>>>>>> load data ", ml.v.Query)
	exec := ml.ctx.(sqlexec.RestrictedSQLExecutor)
	stmt, err := exec.ParseWithParamsInternal(context.Background(), ml.v.Query)
	if err != nil {
		return nil, errors.Errorf("invalid query=%v, err=%v", ml.v.Query, err)
	}
	rows, _, err := exec.ExecRestrictedStmt(context.Background(), stmt)
	if err != nil {
		return nil, errors.Trace(err)
	}

	var labelData []Label
	var imageData []RawImage
	for _, r := range rows {
		vx := r.GetBytes(0)
		imageData = append(imageData, vx)
		vy := r.GetFloat64(1)
		labelData = append(labelData, Label(vy))
	}

	inputs = prepareX(imageData, tensor.Float64)
	targets = prepareY(labelData, tensor.Float64)

	// the data is in (numExamples, 784).
	// In order to use a convnet, we need to massage the data
	// into this format (batchsize, numberOfChannels, height, width).
	//
	// This translates into (numExamples, 1, 28, 28).
	//
	// This is because the convolution operators actually understand height and width.
	//
	// The 1 indicates that there is only one channel (MNIST data is black and white).
	numExamples := inputs.Shape()[0]
	bs := 100
	// todo - check bs not 0

	if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
		log.Fatal(err)
	}
	g := G.NewGraph()
	x := G.NewTensor(g, dt, 4, G.WithShape(bs, 1, 28, 28), G.WithName("x"))
	y := G.NewMatrix(g, dt, G.WithShape(bs, 10), G.WithName("y"))
	m := newConvNet(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	// Note: the correct losses should look like that
	//
	// The losses that are not commented out is used to test the stabilization function of Gorgonia.
	//losses := G.Must(G.HadamardProd(G.Must(G.Neg(G.Must(G.Log(m.out)))), y))

	//losses := G.Must(G.Log(G.Must(G.HadamardProd(m.out, y))))
	//cost := G.Must(G.Mean(losses))
	//cost = G.Must(G.Neg(cost))

	squaredError := must(G.Square(must(G.Sub(m.out, y))))
	cost := must(G.Mean(squaredError))

	// we wanna track costs
	var costVal G.Value
	G.Read(cost, &costVal)

	if _, err = G.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// debug
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	// log.Printf("%v", prog)
	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist())

	prog, locMap, _ := G.Compile(g)
	//log.Printf("%v", prog)

	vm := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap), G.BindDualValues(m.learnables()...))
	solver := G.NewRMSPropSolver(G.WithBatchSize(float64(bs)))
	defer vm.Close()
	// pprof
	// handlePprof(sigChan, doneChan)

	//var profiling bool
	//if *cpuprofile != "" {
	//	f, err := os.Create(*cpuprofile)
	//	if err != nil {
	//		log.Fatal(err)
	//	}
	//	profiling = true
	//	pprof.StartCPUProfile(f)
	//	defer pprof.StopCPUProfile()
	//}
	//go cleanup(sigChan, doneChan, profiling)

	batches := numExamples / bs
	log.Printf("Batches %d", batches)

	epochs := 1
	for i := 0; i < epochs; i++ {
		fmt.Println("Epoch ", i)
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(G.S(start, end)); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			G.Let(x, xVal)
			G.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d, batch %d. Error: %v", i, b, err)
			}
			if err = solver.Step(G.NodesToValueGrads(m.learnables())); err != nil {
				log.Fatalf("Failed to update nodes with gradients at epoch %d, batch %d. Error %v", i, b, err)
			}

			logMaster("Epoch %d | Batch %d | cost %v", i, b, costVal)
			preds := make([]int, 0, bs)
			data := m.out.Value().Data().([]float64)
			//fmt.Printf("shape = %v", m.out.Shape())
			for i := 0; i < bs; i++ {
				idx := -1
				for j := 0; j < 10; j++ {
					if idx == -1 || math.Abs(data[i * 10 + j]) > math.Abs(data[i * 10 + idx]) {
						idx = j
					}
				}
				preds = append(preds, idx)
			}
			actuals := make([]int, 0, bs)
			data = yVal.Data().([]float64)
			//fmt.Printf("actual value = %v", data)
			for i := 0; i < bs; i++ {
				idx := -1
				for j := 0; j < 10; j++ {
					if idx == -1 || math.Abs(data[i * 10 + j]) > math.Abs(data[i * 10 + idx]) {
						idx = j
					}
				}
				actuals = append(actuals, idx)
			}
			cnt := 0
			for i := 0; i < bs; i++ {
				if preds[i] == actuals[i] {
					cnt++
				}
			}
			logMaster("right = %v, acc = %v", cnt, float64(cnt) / float64(bs))

			vm.Reset()
		}
	}

	fileName := "./weights.bin"
	f, err := os.Create(fileName)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	weights := m.learnables()
	vals := make([]G.Value, 0, len(weights))
	for _, w := range weights {
		vals = append(vals, w.Value())
	}
	if err := enc.Encode(vals); err != nil {
		return nil, err
	}

	return []byte(fileName), nil

}

func prepareX(M []RawImage, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(M)
	cols := len(M[0])

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, pixelWeight(M[i][j]))
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < len(M[i]); j++ {
				b = append(b, float32(pixelWeight(M[i][j])))
			}
		}
		backing = b
	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func prepareY(N []Label, dt tensor.Dtype) (retVal tensor.Tensor) {
	rows := len(N)
	cols := 10

	var backing interface{}
	switch dt {
	case tensor.Float64:
		b := make([]float64, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 1.0)
				} else {
					b = append(b, 0.0)
				}
			}
		}
		backing = b
	case tensor.Float32:
		b := make([]float32, rows*cols, rows*cols)
		b = b[:0]
		for i := 0; i < rows; i++ {
			for j := 0; j < 10; j++ {
				if j == int(N[i]) {
					b = append(b, 0.9)
				} else {
					b = append(b, 0.1)
				}
			}
		}
		backing = b

	}
	retVal = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(backing))
	return
}

func pixelWeight(px byte) float64 {
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

const pixelRange = 255
