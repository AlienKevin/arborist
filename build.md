# Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f test.tar.gz tests
```

# Building Docker for Multi-Platform

```
docker buildx build --push --platform linux/amd64,linux/arm64 -t arborist .
```

# Build for mac
```
docker buildx build --load --platform linux/arm64 -t arborist .
```
