# Full version
## Replace include in plot.py
```
include = ['W8T3', 'W308T1', 'W51T1', 'W48T1', 'W156T1', 'W112T1', 'W40T1', 'W56T1', 'W80T1', 'W307T2', 'W295T1', 'W239T2', 'W204T1', 'W110T2', 'W218T1', 'W239T1', 'W307T1', 'W6T1', 'W261T2', 'W253T1', 'W169T1', 'W228T3', 'W228T2', 'W218T2', 'W144T1', 'W120T1', 'W228T1', 'W91T2', 'W268T1', 'W49T1', 'W250T1', 'W303T1', 'W110T3', 'W302T1', 'W8T2', 'W8T1', 'W265T2', 'W265T1', 'W228T4', 'W274T1', 'W87T1', 'W133T1', 'W1T1', 'W138T1', 'W154T1', 'W232T2', 'W240T1', 'W99T1', 'W1T2', 'W284T1', 'W296T1', 'W162T1', 'W309T1', 'W178T1', 'W33T1', 'W263T2', 'W278T1', 'W134T1', 'W158T2', 'W189T1', 'W49T2', 'W34T2', 'W252T1', 'W158T1', 'W252T2', 'W25T1',
           'W287T2', 'W237T1', 'W87T2', 'W34T3', 'W69T1', 'W232T1', 'W77T1', 'W34T1', 'W214T1', 'W226T1', 'W124T1', 'W287T1', 'W74T1', 'W139T1', 'W115T1', 'W304T2', 'W262T1', 'W173T1', 'W74T2', 'W304T1', 'W1T3', 'W164T1', 'W223T1', 'W141T1', 'W305T1', 'W148T1', 'W146T1', 'W233T1', 'W81T1', 'W125T1', 'W146T2', 'W263T1', 'W285T1', 'W190T1', 'W213T1', 'W157T1', 'W157T2', 'W276T1', 'W53T1', 'W127T1', 'W254T1', 'W88T1', 'W69T2', 'W54T2', 'W205T1', 'W91T3', 'W51T2', 'W18T1', 'W188T1', 'W14T1', 'W46T1', 'W52T1', 'W111T2', 'W7T2', 'W9T1', 'W50T1', 'W78T2', 'W111T1', 'W176T1', 'W177T1', 'W149T1', 'W3T1', 'W58T1', 'W149T2', 'W238T1', 'W38T1', 'W78T1', 'W7T1']
include = [x for x in include if x not in ["W7T1", "W51T1", "W78T1"]]
```

## Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f tests.tar.gz tests/benchmarks/W8T3 tests/benchmarks/W308T1 tests/benchmarks/W48T1 tests/benchmarks/W156T1 tests/benchmarks/W112T1 tests/benchmarks/W40T1 tests/benchmarks/W56T1 tests/benchmarks/W80T1 tests/benchmarks/W307T2 tests/benchmarks/W295T1 tests/benchmarks/W239T2 tests/benchmarks/W204T1 tests/benchmarks/W110T2 tests/benchmarks/W218T1 tests/benchmarks/W239T1 tests/benchmarks/W307T1 tests/benchmarks/W6T1 tests/benchmarks/W261T2 tests/benchmarks/W253T1 tests/benchmarks/W169T1 tests/benchmarks/W228T3 tests/benchmarks/W228T2 tests/benchmarks/W218T2 tests/benchmarks/W144T1 tests/benchmarks/W120T1 tests/benchmarks/W228T1 tests/benchmarks/W91T2 tests/benchmarks/W268T1 tests/benchmarks/W49T1 tests/benchmarks/W250T1 tests/benchmarks/W303T1 tests/benchmarks/W110T3 tests/benchmarks/W302T1 tests/benchmarks/W8T2 tests/benchmarks/W8T1 tests/benchmarks/W265T2 tests/benchmarks/W265T1 tests/benchmarks/W228T4 tests/benchmarks/W274T1 tests/benchmarks/W87T1 tests/benchmarks/W133T1 tests/benchmarks/W1T1 tests/benchmarks/W138T1 tests/benchmarks/W154T1 tests/benchmarks/W232T2 tests/benchmarks/W240T1 tests/benchmarks/W99T1 tests/benchmarks/W1T2 tests/benchmarks/W284T1 tests/benchmarks/W296T1 tests/benchmarks/W162T1 tests/benchmarks/W309T1 tests/benchmarks/W178T1 tests/benchmarks/W33T1 tests/benchmarks/W263T2 tests/benchmarks/W278T1 tests/benchmarks/W134T1 tests/benchmarks/W158T2 tests/benchmarks/W189T1 tests/benchmarks/W49T2 tests/benchmarks/W34T2 tests/benchmarks/W252T1 tests/benchmarks/W158T1 tests/benchmarks/W252T2 tests/benchmarks/W25T1 tests/benchmarks/W287T2 tests/benchmarks/W237T1 tests/benchmarks/W87T2 tests/benchmarks/W34T3 tests/benchmarks/W69T1 tests/benchmarks/W232T1 tests/benchmarks/W77T1 tests/benchmarks/W34T1 tests/benchmarks/W214T1 tests/benchmarks/W226T1 tests/benchmarks/W124T1 tests/benchmarks/W287T1 tests/benchmarks/W74T1 tests/benchmarks/W139T1 tests/benchmarks/W115T1 tests/benchmarks/W304T2 tests/benchmarks/W262T1 tests/benchmarks/W173T1 tests/benchmarks/W74T2 tests/benchmarks/W304T1 tests/benchmarks/W1T3 tests/benchmarks/W164T1 tests/benchmarks/W223T1 tests/benchmarks/W141T1 tests/benchmarks/W305T1 tests/benchmarks/W148T1 tests/benchmarks/W146T1 tests/benchmarks/W233T1 tests/benchmarks/W81T1 tests/benchmarks/W125T1 tests/benchmarks/W146T2 tests/benchmarks/W263T1 tests/benchmarks/W285T1 tests/benchmarks/W190T1 tests/benchmarks/W213T1 tests/benchmarks/W157T1 tests/benchmarks/W157T2 tests/benchmarks/W276T1 tests/benchmarks/W53T1 tests/benchmarks/W127T1 tests/benchmarks/W254T1 tests/benchmarks/W88T1 tests/benchmarks/W69T2 tests/benchmarks/W54T2 tests/benchmarks/W205T1 tests/benchmarks/W91T3 tests/benchmarks/W51T2 tests/benchmarks/W18T1 tests/benchmarks/W188T1 tests/benchmarks/W14T1 tests/benchmarks/W46T1 tests/benchmarks/W52T1 tests/benchmarks/W111T2 tests/benchmarks/W7T2 tests/benchmarks/W9T1 tests/benchmarks/W50T1 tests/benchmarks/W78T2 tests/benchmarks/W111T1 tests/benchmarks/W176T1 tests/benchmarks/W177T1 tests/benchmarks/W149T1 tests/benchmarks/W3T1 tests/benchmarks/W58T1 tests/benchmarks/W149T2 tests/benchmarks/W238T1 tests/benchmarks/W38T1 tests/benchmarks/main.rs
```

## Build Docker for Multi-Platform

```
docker buildx build --push --platform linux/amd64,linux/arm64 -t alienkevin/arborist .
```

## Build for testing locally on M1 Mac
```
docker buildx build --load --platform linux/arm64 -t alienkevin/arborist .
```


# Small version

## Replace include in plot.py
```
include = ['W239T1', 'W254T1', 'W14T1', 'W149T1', 'W176T1', 'W296T1', 'W252T1', 'W188T1', 'W287T1', 'W232T1', 'W134T1', 'W164T1', 'W69T1', 'W77T1', 'W33T1', 'W54T2', 'W261T2', 'W173T1', 'W111T2', 'W302T1', 'W214T1', 'W263T1', 'W304T2', 'W228T2', 'W228T4', 'W78T2']
```

## Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f tests.tar.gz tests/benchmarks/W239T1 tests/benchmarks/W254T1 tests/benchmarks/W14T1 tests/benchmarks/W149T1 tests/benchmarks/W176T1 tests/benchmarks/W296T1 tests/benchmarks/W252T1 tests/benchmarks/W188T1 tests/benchmarks/W287T1 tests/benchmarks/W232T1 tests/benchmarks/W134T1 tests/benchmarks/W164T1 tests/benchmarks/W69T1 tests/benchmarks/W77T1 tests/benchmarks/W33T1 tests/benchmarks/W54T2 tests/benchmarks/W261T2 tests/benchmarks/W173T1 tests/benchmarks/W111T2 tests/benchmarks/W302T1 tests/benchmarks/W214T1 tests/benchmarks/W263T1 tests/benchmarks/W304T2 tests/benchmarks/W228T2 tests/benchmarks/W228T4 tests/benchmarks/W78T2 tests/benchmarks/main.rs
```

## Build Docker for Multi-Platform

```
docker buildx build --push --platform linux/amd64,linux/arm64 -t alienkevin/arborist-small .
```

## Build for testing locally on M1 Mac
```
docker buildx build --load --platform linux/arm64 -t alienkevin/arborist-small .
```


# Tiny version

## Replace include in plot.py
```
include = ['W239T1', 'W254T1', 'W14T1', 'W149T1', 'W176T1', 'W296T1', 'W252T1', 'W51T2', 'W78T2', 'W228T4', 'W252T2', 'W1T2']
```

## Zip benchmarks for Docker
```
tar -c --use-compress-program=pigz -f tests.tar.gz tests/benchmarks/W239T1 tests/benchmarks/W254T1 tests/benchmarks/W14T1 tests/benchmarks/W149T1 tests/benchmarks/W176T1 tests/benchmarks/W296T1 tests/benchmarks/W252T1 tests/benchmarks/W51T2 tests/benchmarks/W78T2 tests/benchmarks/W228T4 tests/benchmarks/W252T2 tests/benchmarks/W1T2 tests/benchmarks/main.rs
```

## Build Docker for Multi-Platform

```
docker buildx build --push --platform linux/amd64,linux/arm64 -t alienkevin/arborist-tiny .
```

## Build for testing locally on M1 Mac
```
docker buildx build --load --platform linux/arm64 -t arborist-tiny .
```

# Convert README_source.md to PDF
Use the markdown-pdf extension in vscode.
