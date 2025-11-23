---
title: "HNSW for Vector Databases"
description: "Understanding HNSW (Hierarchical Navigable Small World) algorithm for efficient vector similarity search"
date: 2025-11-20
draft: false
---

While making some vector DB related decisions at work, I tried to find details regarding indexing of vector DBs but I couldn't find a single source that explained HNSW in enough detail, so I thought I'd write a blog which clarifies the basics along with answers to the questions I had!

### What is a vector?
A numerical representation of data (text, images, audio, etc). These are usually high dimension arrays of floats (768 or more) which are the outputs of deep learning models, encoding the meaning of the data into numbers.

### What is a vector DB?
A database with special handling for vector storage, indexing, and retrieval.

### Why index a vector DB? And why not similar to how we index a normal DB?
Vector searches work on similarity, not exact matches. Indexing means nearest-neighbor search faster, instead of comparing to each vector in the entire DB (O(n)). Vector DBs index relationships, not actual values. Making this indexing efficient is important for quick queries.

### What is HNSW?
HNSW stands for Hierarchical Navigable Small World graphs. It implements a high-recall approximate nearest-neighbor (ANN) search for large collections of vectors.

<br>
<div style="text-align: center;">
<img src="/images/HNSW.png" alt="drawing" style="width:800px;"/>

HNSW structure
</div>

HNSW organises vectors into a layered graph where each layer helps narrow down the search, so the query vector is compared to relatively few other vectors. At each layer from top to bottom, the number of nodes increases. This helps us find the approximate region where the similar items may be, refining the region at each layer.

### How are the number of layers decided?
Each vector is assigned to a random layer. Number of vectors per layer increases exponentially as we go down.

### How do we decide which vector goes on which layer?
Random assignment, but each vector appears at its assigned level and on all levels below it. Hence, all vectors are present on Layer 0 (bottom layer).

### How does search work in HNSW?

An easy way to think about it: The higher levels are like highways, then major roads, then streets, then the final address on the bottom layer. The upper levels point you towards the region where the final address is, instead of exploring each street.

In more detail, it works in 2 phases:

**Greedy descent through upper layers:**
Start from a random entry point at the top level. For each layer, find the neighbor closest to the query till no closer neighbor exists, then drop a layer and repeat.

**Best-first search (aka EF-search) at layer 0:**
Here, there are 2 heaps.
1. Candidate queue (min-heap): All the nodes to consider expanding next.
2. Result heap (max-heap): The best possible current candidates, upto size EF (explained below).

The final top-K neighbors come from the result heap.
The time complexity is O(log N).

### What does EF parameter control?
EF stands for exploration factor.
There are 2 EF parameters, EF construction and EF search.

**efConstruction** is used when building the index and for insertion. It controls the number of candidate neighbors considered when a new vector is inserted in the graph. Higher number means a more connected graph (hence better recall (accuracy)), but it slows down index build time and increases memory consumption.

**efSearch** is used during query time. It controls the size of the dynamic candidate list when a query is being run. A higher value means the algorithm explores more potential neighbors while searching. Similarly to efConstruction, a higher value gives higher accuracy, but slower speed.

### How large is the memory footprint for this index?
- Vectors aren't duplicated across layers, links are.
- The main tuning parameter is M (maximum degree of the graph).

M controls the number of connections each node has in the graph.
Graph size is roughly a bit more than ≈ `num_vectors * M * pointer_size`.

### How do deletions work?
The original HNSW paper does not describe a deletion algorithm, and HNSW is generally optimized for append-only or static datasets.

However, practical implementations like **pgvector** handle deletions using a "soft delete" combined with a vacuum process:

1. **Soft Delete**: When a row is deleted, it isn't immediately removed from the HNSW graph. It is just marked as "dead". The graph structure remains intact, and searches still traverse these nodes but ignore them in the final result.
2. **Vacuum / Repair**: To reclaim space and fix the graph, `pgvector` uses a **VACUUM** process (specifically `RepairGraph` pass).

If you don't run VACUUM, the "dead" nodes pile up, wasting memory and potentially slowing down searches (as the algorithm wastes time traversing deleted nodes).

### How do inserts work?
Insertion happens one element at a time.

1. **Assign Layer**: The new element is assigned a maximum layer $l$ based on an exponential distribution (making higher layers scarcer).
2. **Find Entry Point**: Start from the top layer. Greedily traverse the graph to find the closest node to the new element in the current layer. Move down to the next layer using this node as the entry point. Repeat until you reach layer $l$.
3. **Connect Neighbors**: From layer $l$ down to 0:
    - Perform a search to find the $M$ closest neighbors to the new element.
    - Add bidirectional edges between the new element and these neighbors.
    - If a neighbor now has too many connections (exceeding $M_{max}$), prune the connections (usually keeping the closest and most diverse ones) to maintain the graph properties.

### What other options do I have?
IVFFlat is also a commonly used index, which is faster to build and has a smaller memory footprint, but gives lower recall (usually) and is harder to tune.

HNSW gives a more reliable balance of speed and accuracy.

HNSW may not be the best choice if:
- You have a very small dataset (just do exact search).
- You have very frequent writes.
- Memory is very precious.

### Which similarity metric should I use?
Depends on what you are embedding:
- **Cosine similarity**: When vector magnitude doesn't matter as much (good for most language/image embeddings).
- **L2 distance**: When magnitude is important.

### What recall should I expect? Is it fine practically if I don't have 100% recall?
Recall is the ratio of true nearest neighbors that the query successfully returns. To get 100% recall, your queries would be super slow. For most practical use cases, 95%+ recall should be good enough, but it depends on your application.

### Are results reproducible?
Valid question given this is an ANN...

The search results are **deterministic** for a fixed, static graph. If you run the same query twice on the exact same index (and the index hasn't changed), you get the same result.

However, the **index construction is non-deterministic**:
- **Random Layer Assignment**: The layer assignment for each vector is random.
- **Parallelism**: If the index is built using multiple threads (which `pgvector` supports), the order of insertions varies. Since the graph structure depends on the order of insertion, the final graph will differ between runs.

So, if you build the index twice on the same data, you might get slightly different graphs, which leads to slightly different recall or result orderings for the same query.

### Resources
- [Exploring the Internals of pgvector](https://www.linkedin.com/pulse/exploring-internals-pgvector-zhao-song-ynpqf)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
