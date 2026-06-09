# Day 3: LLMs, Embeddings & Vector Databases
### DocuMind Learning Notes

---

## What We Covered Today

Today was the day the "AI magic" started making real sense. We covered three big ideas — how LLMs actually work under the hood, what embeddings are, and how to store and search them with a vector database (Qdrant). We also implemented **hybrid search** (BM25 + vectors) and **reranking** with Cohere, which is the production standard for retrieval quality.

---

## Part 1: How LLMs Actually Work

### The One-Line Truth
> An LLM is a fancy autocomplete. It predicts the next word, one at a time.

That's it. You give it words, it predicts the next word, appends it, predicts the next, and so on until it's done.

### Three Things You Must Believe

| Fact | What It Means For You |
|---|---|
| **LLMs have no memory between calls** | The "memory" in chat apps is a trick — the app resends the entire history every time |
| **LLMs are random** | Same input → different output. Controlled by `temperature` |
| **LLMs don't know facts — they pattern-match** | This is why they hallucinate. And why RAG exists. |

### Temperature — the randomness dial

```
temperature = 0.0   →  always picks the most likely word (predictable, boring)
temperature = 0.5   →  natural, varied output
temperature = 1.2   →  creative, sometimes incoherent
```

**Rule of thumb:**
- RAG, classification, structured output → `0.0` to `0.3`
- Chat, writing → `0.5` to `0.8`
- Brainstorming → `0.8` to `1.2`

### Tokens — the unit of LLM money

LLMs don't see words — they see **tokens** (sub-word chunks).

```python
"hello"        → 1 token
"tokenization" → 2 tokens: ["token", "ization"]
"こんにちは"    → 5 tokens  ← non-English costs more!
```

**Why tokens matter:**
- You pay per token (input + output)
- Output costs ~3–5× more than input
- There's a hard limit per call (128K–1M tokens depending on model)
- Always measure tokens — never guess

---

## Part 2: Embeddings

### The Core Idea
> An embedding is the GPS coordinate of a piece of text on a "meaning map."

An **embedding model** takes any text and converts it to a list of numbers (e.g. 3072 numbers). Texts with similar meaning get similar numbers. Texts with different meaning get very different numbers.

```
embedding("dog")       = [0.12, -0.45, 0.78, ...]
embedding("puppy")     = [0.13, -0.42, 0.81, ...]  ← very close
embedding("airplane")  = [-0.88, 0.91, -0.22, ...]  ← very different
```

### Why This Is Powerful

Old keyword search: "car" only finds documents containing the word "car."

Embedding search: "car" and "automobile" have nearly identical embeddings — so searching for "car" also finds documents about vehicles, sedans, and automobiles. **Meaning matters, not just words.**

### The Golden Rule of Embeddings
> The model you use to **index** documents must be the same one you use to **search**.
> Mixing models = GPS coordinates from different maps = meaningless comparisons.

### What We Used

| Purpose | Model | Dimension |
|---|---|---|
| Embedding documents | `gemini-embedding-001` | 3072 |
| Embedding queries | `gemini-embedding-001` (RETRIEVAL_QUERY mode) | 3072 |

Gemini's embedding model has two modes — always use:
- `RETRIEVAL_DOCUMENT` when indexing (storing docs)
- `RETRIEVAL_QUERY` when searching (user's question)

This improves retrieval quality by ~5–10%.

### The Code

```python
class Embedder:
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = "gemini-embedding-001"
        self.dim = 3072

    async def embed(self, text: str, task_type="RETRIEVAL_QUERY") -> list[float]:
        result = await self.client.aio.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values

    async def embed_many(self, texts: list[str], task_type="RETRIEVAL_DOCUMENT"):
        result = await self.client.aio.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return [e.values for e in result.embeddings]
```

---

## Part 3: Vector Databases — Qdrant

### What a Vector Database Does

A vector database stores embeddings and searches them fast. Three operations:

1. **Insert** — store text + its embedding + metadata
2. **Search** — find the N stored embeddings closest to a query embedding
3. **Filter** — but only within a certain source/category

### Why Not Just Use Postgres?

For small collections (< 100K docs), `pgvector` works fine. For millions of embeddings, you need a database optimized for one job — find nearest neighbours fast. That's Qdrant.

Qdrant uses an algorithm called **HNSW** internally. You don't need to understand it — just know it makes search over millions of vectors nearly instant.

### Setup

```bash
# Run Qdrant locally with Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```

Dashboard: `http://localhost:6333/dashboard`

### Key Qdrant Concepts

| Term | Plain English |
|---|---|
| **Collection** | A table in a regular database |
| **Point** | One row — has an ID, a vector, and a payload |
| **Payload** | Metadata attached to a point (the text, source file, etc.) |
| **Distance.COSINE** | How we measure similarity between vectors |

### Important API Change (learned the hard way!)
`.search()` is **deprecated**. Use `.query_points()` instead:

```python
# OLD (broken in new versions)
results = await client.search(collection_name=..., query_vector=...)

# NEW (correct)
response = await client.query_points(collection_name=..., query=...)
results = response.points  # ← note the .points
```

---

## Part 4: Hybrid Search (BM25 + Vector)

### The Problem With Pure Vector Search

Vector search finds "similar" — but similar ≠ relevant. Example:

- Query: "How do I cancel my subscription?"
- "Our subscription plans include monthly and yearly options" ← similar words, wrong answer
- "To cancel, go to Settings → Billing → Cancel" ← actually relevant

We need both:
- **Vector search** for meaning (catches "automobile" = "car")
- **BM25** for exact keywords (catches product codes, names, error messages)

### What is BM25?

BM25 is keyword search — the mathematical formula behind traditional search engines. It scores documents by:
- Does it contain the query words?
- How many times?
- How rare are those words? (rare words = more signal)
- Is the document short or long?

**BM25 can't understand meaning. Vector search can't catch exact keywords. Together they're stronger than either alone.**

### Hybrid Search in Qdrant — RRF Fusion

Qdrant merges dense (vector) and sparse (BM25) results using **Reciprocal Rank Fusion (RRF)**:

```
Dense results:         Sparse results:
1. Doc A               1. Doc C
2. Doc B               2. Doc A
3. Doc C               3. Doc D

RRF result:
1. Doc A  ← high in BOTH → wins
2. Doc C  ← #1 sparse, #3 dense → strong
3. Doc B  ← dense only
4. Doc D  ← sparse only
```

Formula: `score = 1/(rank + 60)` per list, then sum. High in both = dominant winner.

### Collection Setup for Hybrid

```python
await client.create_collection(
    collection_name="documind",
    vectors_config={
        "dense": VectorParams(size=3072, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(),  # for BM25
    },
)
```

---

## Part 5: Reranking with Cohere

### The Two-Step Pattern

```
Query
  │
  ▼
Vector Search  →  top 20 results  (fast, approximate)
  │
  ▼
Reranker       →  top 5 results   (slower, precise)
  │
  ▼
LLM gets the best 5 chunks
```

### Why Reranking Exists

Vector search encodes query and document **separately** then compares. The query and document never "see" each other during encoding.

A **cross-encoder** (what Cohere uses) reads the query and each document **together**. It can see how well they interact. Much more accurate — but too slow to run over your entire database. Hence: fetch 20 fast candidates, rerank to get the best 5.

### Analogy
- Vector search = fast librarian grabbing 20 roughly-relevant books
- Reranker = careful scholar reading all 20 and picking the best 5

### Cohere Rerank Code

```python
response = await self.client.rerank(
    model="rerank-v3.5",
    query=query,
    documents=[r.chunk.text for r in results],
    top_n=5,
    return_documents=False,
)

# Cohere returns indices into your original list + new scores
reranked = [
    SearchResult(chunk=results[hit.index].chunk, score=hit.relevance_score)
    for hit in response.results
]
```

---

## Part 6: Vectorless Search

### When You Don't Need a Vector Database

| Collection Size | Approach | Why |
|---|---|---|
| < 50 pages | **Full context** — stuff everything in | Simple, accurate, cheap |
| 50–500 pages | **LLM-as-retriever** — LLM decides what's relevant | Most accurate, more expensive |
| 500+ pages | **Hybrid + Rerank** (what DocuMind uses) | Only scalable option |

### Full Context (Approach A)

```python
# No retrieval. Just give the LLM ALL documents and ask it to answer.
prompt = f"""
Documents:
{all_docs}

Question: {question}
Answer based ONLY on the documents above.
"""
```

Simple. Zero infrastructure. Breaks down when docs exceed the context window.

### LLM-as-Retriever (Approach B)

```python
# Ask a cheap model: "Is this doc relevant? YES or NO"
# Only pass YES docs to the final answer step
```

Most accurate relevance filtering. Very expensive — LLM call per document.

---

## The Full DocuMind Retrieval Pipeline (After Today)

```
User query
    │
    ▼
Embedder.embed(query)             ← Gemini, RETRIEVAL_QUERY
    │
    ├── dense vector ──────────────┐
    │                              ├── Qdrant RRF fusion → 20 candidates
    └── sparse vector (BM25) ─────┘
                │
                ▼
    Reranker.rerank(query, 20 candidates)   ← Cohere cross-encoder
                │
                ▼
          Top 5 truly relevant chunks
                │
                ▼
          (Day 4) → LLM answers with citations
```

---

## Errors We Hit and Fixed

| Error | Root Cause | Fix |
|---|---|---|
| `404 NOT_FOUND: text-embedding-004` | Model deprecated by Google | Switch to `gemini-embedding-001` |
| `chunks and embeddings must be same length` | `gemini-embedding-2` returns ONE aggregated vector for a batch, not one per text | Stick with `gemini-embedding-001` which batches correctly |
| `AttributeError: has no attribute 'search'` | Qdrant deprecated `.search()` | Replace with `.query_points()` and access `.points` on the result |

**The production lesson:** AI library APIs change fast. Always read errors literally — they tell you exactly what broke. Validate at boundaries so failures are loud, not silent.

---

## Model Selection Framework

| Question | Why It Matters |
|---|---|
| Does cheapest model work well enough? | Always test cheap first |
| What's my latency budget? | Small/fast models for < 1s responses |
| How big is my context? | Most are 128K+; Gemini goes to 1M |
| Do I need structured output? | Verify reliability per model |
| Privacy/compliance needs? | Local model if data can't leave your servers |

**DocuMind's choices:**
- Chat: Groq Llama 3.3 70B (fast + your key)
- Embeddings: Gemini `gemini-embedding-001` (free + batches correctly)
- Reranking: Cohere `rerank-v3.5` (cross-encoder accuracy)
- Fallback chat: Gemini 2.0 Flash

---

## Key Takeaways

1. **LLMs are stateless autocomplete** — no magic, just very good next-word prediction
2. **Embeddings = GPS for meaning** — similar text → similar coordinates
3. **Vector search finds similar; reranking finds relevant** — they're not the same thing
4. **Hybrid search (BM25 + vector) is the production standard** — better than either alone
5. **Vectorless is valid for small collections** — don't over-engineer when 50 docs fit in context
6. **Loud failures > silent corruption** — always validate lengths/shapes at boundaries
7. **API names change** — read errors literally, fix the specific thing, move on

---

## Files Added to DocuMind Today

```
src/documind/
├── embeddings.py      ← Gemini embedding wrapper (embed + embed_many)
├── vectorstore.py     ← Qdrant with hybrid search (dense + sparse + RRF)
├── reranker.py        ← Cohere rerank wrapper
├── retriever.py       ← Full pipeline: embed → hybrid search → rerank
└── vectorless_search.py  ← Two vectorless approaches for small collections

scripts/
├── 03_test_vectorstore.py   ← Smoke test: store + search
└── 03b_test_retriever.py    ← Full pipeline test with reranking
```

---

*Next up → Day 4: RAG, LangChain & Evals — where these retrieved chunks become cited answers.*
