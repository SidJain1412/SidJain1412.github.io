---
title: "HNSW for Vector Databases Explained"
description: "Understanding HNSW (Hierarchical Navigable Small World) algorithm for efficient vector similarity search"
date: 2025-12-08
draft: false
---

While making some vector DB related decisions at work, I tried to find details regarding indexing of vector DBs, but I couldn't find a single source that explained HNSW in enough detail. So I thought I'd write a short blog post that clarifies the basics along with answers to the questions I had!

### What is a vector?
A numerical representation of data (text, images, audio, etc.). At a simple level, you can think of a vector as an ordered list of numbers like `[0.2, -1.3, 4.5]` that a computer uses to describe something.

In machine learning, these numbers are usually high-dimensional arrays of floats (often 768 dimensions or more). They are produced by deep learning models which learn to encode different aspects of the input (topic, style, semantics, etc.) into numbers, so that **similar things end up with similar vectors**.

You don't usually interpret individual dimensions by hand. Instead, you compare vectors (using cosine similarity or L2 distance) to see which pieces of data are closest in this learned space (i.e. similar to each other)

Find a more detailed explanation [**here**](https://cloud.google.com/blog/topics/developers-practitioners/meet-ais-multitool-vector-embeddings)

<br>
<div style="text-align: center;">
<img src="/images/vectors_google.png" alt="vectors" style="width:600px;"/>

Similar words get grouped similarly by good embedding models
</div>

### What is a vector DB?
A database with special handling for vector storage, indexing, and retrieval. In other words, it’s a database built to store these long number lists (embeddings) and quickly find “similar” ones.

### How does vector similarity work? (simple example)
Imagine you have three short texts:
- A: "I love trekking"
- B: "Hiking is enjoyable in monsoon"
- C: "The stock market isn't doing well"

A model turns each of these into a vector. Texts A and B end up with vectors that point in a similar “direction” (because they are both about outdoor activities), while C points in a very different direction (because it’s about finance).

When a user searches for "good time for trekking", we turn the query into a vector and compare it to all document vectors. A and B have high similarity scores, C has a low score, so we return A and B. This is why vector similarity search is useful: it can match based on *meaning*, not just exact words.

### Why index a vector DB? And why not index it similar to how we index a normal DB?
Vector searches work on similarity, not exact matches. Indexing is just building a side data structure (like the index of a book) that makes nearest-neighbor search faster, instead of comparing to each vector in the entire DB (O(n)).

If you only have a few thousand vectors, scanning all of them for each query is usually fine. But once you have millions of vectors and many queries per second, an O(n) scan for every query becomes too slow and expensive. At that point you need an approximate nearest neighbor index like HNSW to get "good enough" results much faster.

Vector DBs index relationships, not actual values. Making this indexing efficient is important for quick queries.

### What is HNSW?
HNSW stands for **Hierarchical Navigable Small World** graphs. It implements a high-recall approximate nearest-neighbor (ANN) search for large collections of vectors. I know the name is a mouthful but it'll be clearer by the end of this post!

<br>
<div style="text-align: center;">
<img src="/images/HNSW.png" alt="drawing" style="width:800px;"/>

HNSW structure
</div>

An easy way to think about it: The higher levels are like highways, then major roads, then streets, then the final address on the bottom layer. The upper levels point you towards the region where the final address is, instead of exploring each street. Just that the assignment of each point as a highway, road, or street is random (more details below).

HNSW organizes vectors into a layered graph where each layer helps narrow down the search, so the query vector is compared to relatively few other vectors.

At each layer from top to bottom, the number of nodes increases. This helps us find the approximate region where the similar items may be, refining the region at each layer.

### What is a "small world"?
The **small world** idea comes from social networks and the classic “[six degrees of separation](https://en.wikipedia.org/wiki/Small-world_experiment)” concept: even in a very large network, you can reach almost anyone with just a few hops.

In graph terms, a small-world network has:
- **Short paths** between most pairs of nodes (you only need a few steps to get from one point to another), and
- **Local clustering**, where neighbors of a node are often connected to each other.


<br>
<div style="text-align: center;">
  <img src="/images/SmallWorld.png" style="width:400px;"/>
  <br>
  6 Degrees of Separation: Any 2 people in the USA can be linked by a chain of <= 6 people
  <a href="https://en.wikipedia.org/wiki/Small-world_experiment">
    (source)
  </a>
</div>



HNSW uses this property so that, starting from almost any node, a greedy walk (always moving to the neighbor that is currently closest) over the graph can very quickly get close to the region where your nearest neighbors live, without having to visit every node.

### How are the HNSW layers built (and why)?
- **Chance of being in higher layers:** When a new vector is added, it is **always** placed on layer 0. Then, for each higher layer (1, 2, 3, …), we randomly decide whether it also appears there. Fewer and fewer vectors appear as we go up in the layers.
- **All points on the bottom:** Because every vector is on layer 0, that layer (the "streets") always contains the full dataset, so you can always do a precise local search there if needed.
- **Local connections by similarity:** On each layer, we connect a point only to its nearest neighbors. We never do a global "find best connections" step.
- **Why this works:** Higher layers have very few points, so they act like long‑distance shortcuts (highways). Lower layers have many points, so once you are in roughly the right area, you can walk the dense local graph to find exact neighbors. The top layers are simply made up of those few points that also got placed in higher layers and then connected to their closest neighbors there, so they naturally become good shortcuts without any extra planning.

### How does search work in HNSW?
The highway analogy explained it shortly, but in more detail, it works in 2 phases:

**Step 1: Walk down through the coarse layers (greedy descent):**
Start from a random entry point at the top level. For each layer, repeatedly move to the neighbor closest to the query until no closer neighbor exists, then drop a layer and repeat.

**Step 2: Carefully explore around the best candidates (best-first search / EF-search) at layer 0:**
Here, there are 2 small ranked lists (priority queues) that are used only during the search process (the actual vectors stay in the index graph):
1. **Candidate queue**: Nodes we might want to expand next (potentially good neighbors whose neighbors we haven't explored yet). The algorithm always picks the currently closest candidate from this list to explore further.
2. **Result list**: The best current candidates (nearest neighbors found so far), kept up to size **EF** (explained below) and roughly sorted from closest to farthest.

The final top-K neighbors come from the result heap.

In practice, the number of nodes explored grows slowly with the dataset size because of the layered small-world structure, so the search behaves roughly like **O(log N)** (though this is empirical, not a strict guarantee).

### What does EF parameter control?
EF stands for exploration factor.
There are 2 EF parameters: **EF construction** and **EF search**.

**ef_construction** is used when building the index and for insertion. It controls the number of candidate neighbors considered when a new vector is inserted in the graph (how widely we look around when we add a new point).

Higher values mean a more connected graph (hence better recall/accuracy), but they slow down index build time and increase memory consumption.

**efSearch** is used during query time. It controls the size of the dynamic candidate list (the result heap and candidate queue) when a query is being run (how many extra candidates we are willing to examine).

A higher value means the algorithm explores more potential neighbors while searching. Similar to efConstruction, a higher value gives higher accuracy, but slower speed.

### How do inserts work?
Insertion happens one element at a time.

1. **Assign Layer**: The new element is assigned a maximum layer $l$ (higher layers are scarcer; most points only appear in lower layers).
2. **Find Entry Point**: Start from the top layer. Greedily traverse the graph to find the closest node to the new element in the current layer.
   Move down to the next layer using this node as the entry point. Repeat until you reach layer $l$.
3. **Connect Neighbors**: From layer $l$ down to 0:
    - Perform a search to find the $M$ closest neighbors to the new element.
    - Add bidirectional edges (connections) between the new element and these neighbors.
    - If a neighbor now has too many connections (exceeding $M_{max}$), prune the connections (usually keeping the closest and most diverse ones) to maintain the graph properties.

### How do deletions work?
The original HNSW paper does not describe a deletion algorithm, and HNSW is generally optimized for append-only or static datasets.

However, practical implementations like **pgvector** handle deletions using a "soft delete" combined with a cleanup / vacuum process:

1. **Soft Delete**: When a row is deleted, it isn't immediately removed from the HNSW graph. It is just marked as "dead" (logically deleted). The graph structure remains intact, and searches still traverse these nodes but ignore them in the final result.
2. **Vacuum / Repair**: To reclaim space and fix the graph, `pgvector` uses a **VACUUM** process (specifically `RepairGraph` pass) — you can think of this as a background cleanup that actually removes dead nodes and reconnects their neighbors.

If you don't run VACUUM, the "dead" nodes pile up, wasting memory and potentially slowing down searches (as the algorithm wastes time traversing deleted nodes).

### Are results reproducible?
The search results are **deterministic** for a fixed, static graph. If you run the same query twice on the exact same index (and the index hasn't changed), you get the same result.

However, the **index construction is non-deterministic** (two index builds on the same data can give slightly different graphs):
- **Random Layer Assignment**: The layer assignment for each vector is random.
- **Parallelism**: If the index is built using multiple threads (which `pgvector` supports), the order of insertions varies. Since the graph structure depends on the order of insertion, the final graph will differ between runs.

So, if you build the index twice on the same data, you might get slightly different graphs, which leads to slightly different recall or result orderings for the same query.

### How large is the memory footprint for this index?
- Vectors aren't duplicated across layers, links are.
- The main tuning parameter is **M** (maximum degree of the graph).

M controls the number of connections each node has in the graph (how many neighbors each point can link to).
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


<br>
<div style="text-align: center;">
  <img src="/images/GraphConnections.png" alt="Graph with decreasing connectivity (lower M) from left to right" style="width:600px;"/>
  <br>
  Decreasing M from left to right
  <a href="https://inviqa.com/blog/storing-graphs-database-sql-meets-social-network">
    (source)
  </a>
</div>

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

<br>
<div style="text-align: center;">
  <img src="/images/SimMetrics.png"style="width:600px;"/>
  <br>
  <a href="https://www.linkedin.com/pulse/math-similarity-cohesion-manu-nellutla/">
    (source)
  </a>
</div>


### What other options do I have?
[IVFFlat](https://www.tigerdata.com/blog/nearest-neighbor-indexes-what-are-ivfflat-indexes-in-pgvector-and-how-do-they-work) is also a commonly used index, which is faster to build and has a smaller memory footprint, but gives lower recall (usually) and is harder to tune.

HNSW gives a more reliable balance of speed and accuracy.

HNSW may not be the best choice if:
- You have a very small dataset (just do exact search).
- You have very frequent writes.
- Memory is very precious.

### Conclusion
HNSW gives us a practical way to do fast, “good enough” nearest-neighbor search on huge collections of embeddings, without scanning everything on every query. By organizing points into a small-world, multi-layer graph, it lets us jump quickly across the space on sparse upper layers and then refine locally on the dense bottom layer. 

With just a few knobs like **M** and **efSearch**, you can trade off speed, memory, and recall as per your application’s needs.

### Resources
- [Exploring the Internals of pgvector](https://www.linkedin.com/pulse/exploring-internals-pgvector-zhao-song-ynpqf)
- [Pinecone HNSW Guide](https://www.pinecone.io/learn/series/faiss/hnsw/)
- [Understanding how Vector DBs work](https://youtu.be/035I2WKj5F0?si=EMS8Mo2ahsUQGwh2)
