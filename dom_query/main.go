package main

import (
	"C"

	"github.com/antchfx/htmlquery"
	"github.com/antchfx/xpath"
	"golang.org/x/net/html"
)
import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
)

var benchmarkName string
var docId uint16
var doc *html.Node

//export SetupBenchmark
func SetupBenchmark(name string) {
	// debug.SetGCPercent(100)
	// runtime.GC()

	fmt.Println("SetupBenchmark: " + name)
	benchmarkName = name

	htmlquery.DisableSelectorCache = true

	// debug.SetGCPercent(-1)
}

//export LoadDoc
func LoadDoc(id uint16) {
	if docId != id {
		docId = id
		newDoc, err := htmlquery.LoadDoc(fmt.Sprintf("%s/doms/%d.html", benchmarkName, docId))
		if err != nil {
			panic(err)
		}
		doc = newDoc
	}
}

//export QuerySelector
func QuerySelector(selector string) bool {
	n, err := xpath.Compile(selector)
	if err != nil {
		panic(err)
	}
	res := htmlquery.QuerySelector(doc, n)
	return res != nil
}

//export QuerySelectors
func QuerySelectors(selectors []string) bool {
	ctx, cancel := context.WithCancel(context.Background()) // use context to terminate other running goroutines
	hasValidSelector := false

	var wg sync.WaitGroup

	wg.Add(len(selectors))
	for _, selector := range selectors {
		go func(sel string) {
			defer wg.Done()
			select {
			case <-ctx.Done():
				return
			default:
				res, err := htmlquery.Query(doc, sel)
				if err != nil {
					panic(err)
				}
				if res != nil {
					hasValidSelector = true
					cancel()
				}
				return
			}
		}(selector)
	}

	wg.Wait()
	return hasValidSelector
}

func main() {
	// do nothing
}
