# Wasp 
Wasp is an asynchronous algorithm for paralell Single-Source Shortest Path.
The Wasp codebase is built on top of the [GAP Benchmarking Suite](https://github.com/sbeamer/gapbs).

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
- `-f graph.el` loads graph from file `graph.el`
- `-sf graph.el` symmetrizes graph loaded from file `graph.el`

Synthetic graphs can be generated through the options:
- `-g 20` generates a Kronecker graph with 2^20 vertices (Graph500 specifications)
- `-u 20` generates an Erdős–Rényi random graph with 2^20 vertices

The graph loading infrastructure understands the following formats (no additional flags are required):

- `.el` plain-text edge-list with an edge per line as `node1 node2`
- `.tsv` plain-text weighted edge-list with an edge per line as `node1 node2 weight` (used in the Konect graph repository)
- `.gr` 9th DIMACS Implementation Challenge format
- `.graph` Metis format (used in 10th DIMACS Implementation Challenge)
- `.mtx` Matrix Market format
- `.sg` serialized pre-built graph
- `.wsg` weighted serialized pre-built graph


To run Wasp on a sample `graph.el` edge-list directed graph, run:
```bash
# in wasp/build
./sssp -f graph.el -n $trials -d $delta 
```
where `$trials` is the number of trials, `$delta` is the value of delta for the delta-stepping algorithm.
The `-R` option can be used to use the same starting vertex for all trials.
Use `sssp -h` to see all options.
