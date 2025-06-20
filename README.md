# GANNS
## Update
We have refactored the project to improve code readability.

## Introduction
This project includes:

(1) GANNS, a GPU-based algorithm that accelerates approximate nearest neighbor (ANN) search on proximity graphs by redesigning classical CPU-based search methods and adopting GPU-friendly data structures;

(2) novel GPU-based construction algorithms for proximity graphs that ensure high-quality graph structures.

## Usage
To use this project, specify the distance metric and data dimensionality in lines 23–24 of ```CMakeLists.txt```.
```zsh
target_compile_definitions(${PROJECT_NAME} PRIVATE [YOUR METRIC])
target_compile_definitions(${PROJECT_NAME} PRIVATE PLACE_HOLDER_DIM=[YOUR DIMENSION])
```
Currently, the project supports arbitrary dimensions and three distance metrics: ```Euclidean distance (USE_DIST_L2_)```, ```cosine similarity (USE_DIST_CS_)```, and ```inner product (USE_DIST_IP_)```.

### Construction
The construction algorithm requires the following parameters to be provided.
```zsh
./GANNS -b [base_path] [graph_type] [d_min] [M] [graph_path]
```
Specifically:

	[base_path] specifies the directory containing the data points (i.e., the database).
 
	[graph_type] defines the type of proximity graph to construct.
 
	[d_min] indicates the minimum degree in the proximity graph (with d_max automatically set to 2 * d_min).
 
	[M] denotes the number of nearest neighbors retained during the search for each node’s neighbors (with e automatically set to M).
 
	[graph_path] specifies the directory where the constructed graph will be saved.

At present, two types of proximity graphs are supported: ```NSW (nsw)``` and ```HNSW (hnsw)```.

For example, the following command builds a NSW graph on the SIFT dataset.

(Note: The files referenced in the command are not included in this project due to their large size.)
```zsh
./GANNS -b dataset/sift/base.fvecs nsw 16 64 idx/sift_16_64.nsw
```

### Search
The search algorithm requires the following parameters to be provided.
```zsh
./GANNS -q [base_path] [query_path] [graph_type] [graph_path] [groundtruth_path] [k] [e] [M]
```
Specifically:

	[base_path] specifies the directory containing the data points (i.e., the database).

	[query_path] specifies the directory containing the query points.
  
	[graph_type] indicates the type of proximity graph to be used.

	[groundtruth_path] is the directory of the ground-truth nearest neighbors.

	[k] denotes the number of nearest neighbors to return.

	[e] specifies the number of vertices to explore during the search.

	[M] denotes the number of nearest neighbors retained during the search for each node’s neighbors (Note: e <= M).

For example, the following command performs approximate nearest neighbor (ANN) search on an NSW graph built from the SIFT dataset.

(Note: The files referenced in the command are not included in this project due to their large size.)
```zsh
./GANNS -q dataset/sift/base.fvecs dataset/sift/query.fvecs nsw idx/sift_16_64.nsw dataset/sift/groundtruth.ivecs 10 64 64
```

## Datasets
The information of used real-world datasets is provided in our paper. Currently, we support ```.fvecs``` file for base datasets and query datasets 
and ```.ivecs``` for groundtruth file.
