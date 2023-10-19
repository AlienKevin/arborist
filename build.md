# Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f tests.tar.gz tests/benchmarks/W239T1 tests/benchmarks/W254T1 tests/benchmarks/W14T1 tests/benchmarks/W149T1 tests/benchmarks/W176T1 tests/benchmarks/W296T1 tests/benchmarks/W252T1 tests/benchmarks/W51T2 tests/benchmarks/W78T2 tests/benchmarks/W228T4 tests/benchmarks/W252T2 tests/benchmarks/W1T2 tests/benchmarks/main.rs
```

# Build Docker for Multi-Platform

```
docker buildx build --push --platform linux/amd64,linux/arm64 -t alienkevin/arborist-small .
```

# Build for testing locally on M1 Mac
```
docker buildx build --load --platform linux/arm64 -t arborist-small .
```

# Convert README_source.md to PDF
Use the markdown-pdf extension in vscode.
