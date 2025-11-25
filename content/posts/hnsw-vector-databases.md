---
title: "HNSW for Vector Databases Explained"
description: "Understanding HNSW (Hierarchical Navigable Small World) algorithm for efficient vector similarity search"
date: 2025-11-23
draft: false
---

While making some vector DB related decisions at work, I tried to find details regarding indexing of vector DBs, but I couldn't find a single source that explained HNSW in enough detail. So I thought I'd write a short blog post that clarifies the basics along with answers to the questions I had!

### What is a vector?
A numerical representation of data (text, images, audio, etc.). At a simple level, you can think of a vector as an ordered list of numbers like `[0.2, -1.3, 4.5]`.

In machine learning, these numbers are usually high-dimensional arrays of floats (often 768 dimensions or more). They are produced by deep learning models which learn to encode different aspects of the input (topic, style, semantics, etc.) into numbers, so that **similar things end up with similar vectors**.

You don't usually interpret individual dimensions by hand. Instead, you compare vectors (using cosine similarity or L2 distance) to see which pieces of data are closest in this learned space (i.e. similar to each other)

Find a more detailed explanation [**here**](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings)

### What is a vector DB?
A database with special handling for vector storage, indexing, and retrieval.

### Why index a vector DB? And why not index it similar to how we index a normal DB?
Vector searches work on similarity, not exact matches. Indexing makes nearest-neighbor search faster, instead of comparing to each vector in the entire DB (O(n)).

Vector DBs index relationships, not actual values. Making this indexing efficient is important for quick queries.

### What is HNSW?
HNSW stands for **Hierarchical Navigable Small World** graphs. It implements a high-recall approximate nearest-neighbor (ANN) search for large collections of vectors. I know the name is a mouthful but it'll be clearer by the end of this post!

<br>
<div style="text-align: center;">
<img src="/images/HNSW.png" alt="drawing" style="width:800px;"/>

HNSW structure
</div>

HNSW organizes vectors into a layered graph where each layer helps narrow down the search, so the query vector is compared to relatively few other vectors.

At each layer from top to bottom, the number of nodes increases. This helps us find the approximate region where the similar items may be, refining the region at each layer.

### What is a "small world"?
The **small world** idea comes from social networks and the classic “six degrees of separation” concept: even in a very large network, you can reach almost anyone with just a few hops.

In graph terms, a small-world network has:
- **Short paths** between most pairs of nodes (you only need a few steps to get from one point to another), and
- **Local clustering**, where neighbors of a node are often connected to each other.

HNSW uses this property so that, starting from almost any node, a greedy walk over the graph can very quickly get close to the region where your nearest neighbors live, without having to visit every node.

### How are the number of layers decided?
Each vector is assigned to a random layer. The number of vectors per layer increases exponentially as we go down.

### How do we decide which vector goes on which layer?
Random assignment, but each vector appears at its assigned level and on all levels below it. Hence, all vectors are present on Layer 0 (bottom layer).

### How does search work in HNSW?
An easy way to think about it: The higher levels are like highways, then major roads, then streets, then the final address on the bottom layer. The upper levels point you towards the region where the final address is, instead of exploring each street.

In more detail, it works in 2 phases:

**Greedy descent through upper layers:**
Start from a random entry point at the top level. For each layer, find the neighbor closest to the query till no closer neighbor exists, then drop a layer and repeat.

**Best-first search (aka EF-search) at layer 0:**
Here, there are 2 heaps that are used only during the search process (the actual vectors stay in the index graph):
1. **Candidate queue (min-heap)**: Nodes we might want to expand next (potentially good neighbors whose neighbors we haven't explored yet).
2. **Result heap (max-heap)**: The best current candidates (nearest neighbors found so far), up to size **EF** (explained below).

The final top-K neighbors come from the result heap.

In practice, the number of nodes explored grows slowly with the dataset size because of the layered small-world structure, so the search behaves roughly like **O(log N)** (though this is empirical, not a strict guarantee).

### What does EF parameter control?
EF stands for exploration factor.
There are 2 EF parameters: **EF construction** and **EF search**.

**efConstruction** is used when building the index and for insertion. It controls the number of candidate neighbors considered when a new vector is inserted in the graph.

Higher values mean a more connected graph (hence better recall/accuracy), but they slow down index build time and increase memory consumption.

**efSearch** is used during query time. It controls the size of the dynamic candidate list (the result heap and candidate queue) when a query is being run.

A higher value means the algorithm explores more potential neighbors while searching. Similar to efConstruction, a higher value gives higher accuracy, but slower speed.

### How do inserts work?
Insertion happens one element at a time.

1. **Assign Layer**: The new element is assigned a maximum layer $l$ based on an exponential distribution (making higher layers scarcer).
2. **Find Entry Point**: Start from the top layer. Greedily traverse the graph to find the closest node to the new element in the current layer.
   Move down to the next layer using this node as the entry point. Repeat until you reach layer $l$.
3. **Connect Neighbors**: From layer $l$ down to 0:
    - Perform a search to find the $M$ closest neighbors to the new element.
    - Add bidirectional edges between the new element and these neighbors.
    - If a neighbor now has too many connections (exceeding $M_{max}$), prune the connections (usually keeping the closest and most diverse ones) to maintain the graph properties.

### How do deletions work?
The original HNSW paper does not describe a deletion algorithm, and HNSW is generally optimized for append-only or static datasets.

However, practical implementations like **pgvector** handle deletions using a "soft delete" combined with a vacuum process:

1. **Soft Delete**: When a row is deleted, it isn't immediately removed from the HNSW graph. It is just marked as "dead". The graph structure remains intact, and searches still traverse these nodes but ignore them in the final result.
2. **Vacuum / Repair**: To reclaim space and fix the graph, `pgvector` uses a **VACUUM** process (specifically `RepairGraph` pass).

If you don't run VACUUM, the "dead" nodes pile up, wasting memory and potentially slowing down searches (as the algorithm wastes time traversing deleted nodes).

### Are results reproducible?
The search results are **deterministic** for a fixed, static graph. If you run the same query twice on the exact same index (and the index hasn't changed), you get the same result.

However, the **index construction is non-deterministic**:
- **Random Layer Assignment**: The layer assignment for each vector is random.
- **Parallelism**: If the index is built using multiple threads (which `pgvector` supports), the order of insertions varies. Since the graph structure depends on the order of insertion, the final graph will differ between runs.

So, if you build the index twice on the same data, you might get slightly different graphs, which leads to slightly different recall or result orderings for the same query.

### How large is the memory footprint for this index?
- Vectors aren't duplicated across layers, links are.
- The main tuning parameter is **M** (maximum degree of the graph).

M controls the number of connections each node has in the graph.
Graph size is roughly a bit more than ≈ `num_vectors * M * pointer_size`.

### How do I choose M? Can I change it later?
You typically configure **M** when you **create** the index (for example, as a parameter when you define the HNSW index in `pgvector` or another vector DB).

Higher **M** means:
- More links per node → denser graph.
- Better recall and more robustness (easier to “route around” bad connections or deletions).
- Higher memory usage and slower index builds/inserts, and slightly slower queries.

Lower **M** means:
- Fewer links per node → sparser graph.
- Less memory and faster inserts/builds.
- Lower recall, especially for harder queries.

Once an index is built, **M is effectively fixed** for that index. To change it, you usually need to **rebuild the index** with a new value of M (recreate the index and reinsert the data).

### What recall should I expect? Is it fine practically if I don't have 100% recall?
Recall is the ratio of true nearest neighbors that the query successfully returns. To get 100% recall, your queries would be super slow.

For most practical use cases, 95%+ recall should be good enough, but it depends on your application.

### Which similarity metric should I use?
Depends on what you are embedding and whether the **magnitude (length)** of the vector carries meaning.

1. **Cosine similarity**: Looks at the angle between vectors and ignores magnitude (it effectively normalizes vectors).
    - Good when you care about **direction** (semantic meaning) but not strength.
    - Typical for language/image embeddings, which are often normalized by the model.

2. **L2 distance**: Looks at full Euclidean distance, so both direction and magnitude matter.
    - Use this when the **scale** of the embedding is meaningful:
    - E.g. models where higher-magnitude vectors mean “stronger” signals or higher confidence.
    - Numeric or feature-engineered vectors where absolute values and ranges are important.

### What other options do I have?
IVFFlat is also a commonly used index, which is faster to build and has a smaller memory footprint, but gives lower recall (usually) and is harder to tune.

HNSW gives a more reliable balance of speed and accuracy.

HNSW may not be the best choice if:
- You have a very small dataset (just do exact search).
- You have very frequent writes.
- Memory is very precious.

### Resources
- [Exploring the Internals of pgvector](https://www.linkedin.com/pulse/exploring-internals-pgvector-zhao-song-ynpqf)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Understanding how Vector DBs work](https://youtu.be/035I2WKj5F0?si=EMS8Mo2ahsUQGwh2)
