test = cargo test --release --test=$(1) -- --nocapture --color=always

test-opt = RUSTFLAGS="-C target-cpu=native" cargo test --profile release-opt --test=$(1) -- --nocapture --color=always

sparsity_experiment: build
	$(call test,sparsity_experiment) > output_sparse.txt

scale_experiment: build
	$(call test,scale_experiment) > output.txt

experiment: build
	$(call test,experiment) > output.txt

experiment-opt: build
	$(call test-opt,experiment) > output.txt

main: build
	$(call test,main) > output.txt

main-opt: build
	$(call test-opt,main) > output.txt

verify: build
	$(call test,verify)

flamegraph: build
	cargo flamegraph --test=main --root -- --color always --nocapture

GO_MAIN := dom_query/main.go
GO_TARGET := dom_query/libdom_query.a

$(GO_TARGET): $(GO_MAIN)
	go fmt $(GO_MAIN)
	CGO_ENABLED=1 go build -buildmode=c-archive -o $(GO_TARGET) $(GO_MAIN)
	bindgen dom_query/libdom_query.h -o src/ffi.rs

# Src: https://github.com/rust-lang/cargo/issues/3591#issuecomment-673356426
build: $(GO_TARGET)
	cargo build --release 2>&1 | grep -Ei "error|aborting|warnings"
