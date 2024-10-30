# Wasp 
Wasp is an asynchronous concurrent priority scheduler for ordered graph analytics.
Wasp is built on top of the [GAP Benchmarking Suite](https://github.com/sbeamer/gapbs) codebase.

## Configuration
Recommended environment:
- C++17 compiler with OpenMP support
- Cmake >= 3.18

Sample configuration and building:
```bash
mkdir build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Running Wasp
All of the binaries use the same command-line options for loading graphs:

- `-g 20` generates a Kronecker graph with 2^20 vertices (Graph500 specifications)
- `-u 20` generates an Erdős–Rényi random graph with 2^20 vertices
- `-f graph.el` loads graph from file graph.el
- `-sf graph.el` symmetrizes graph loaded from file graph.el

The graph loading infrastructure understands the following formats:

- `.el` plain-text edge-list with an edge per line as `node1 node2`
- `.wel` plain-text weighted edge-list with an edge per line as `node1 node2 weight`
- `.gr` 9th DIMACS Implementation Challenge format
- `.graph` Metis format (used in 10th DIMACS Implementation Challenge)
- `.mtx` Matrix Market format
- `.sg` serialized pre-built graph
- `.wsg` weighted serialized pre-built graph


To run Wasp on a sample `graph.el` edge-list graph, run:
```bash
# in wasp/build
./sssp-wasp -f graph.el -n $trials -d $delta 
```
where `$trials` is the number of trials, `$delta` is the value of delta for the delta-stepping algorithm.
The `-R` option can be used to use the same starting vertex for all trials.

