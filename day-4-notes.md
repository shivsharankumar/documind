# Day 4: RAG — Making DocuMind Actually Answer

## Where we are

Day 3 gave DocuMind the ability to **find** relevant chunks (hybrid search + reranking). But finding is only half of RAG. Right now if you search "what is RAG?", you get back the *chunk of text* — not an *answer*. A user doesn't want a chunk. They want a sentence that answers their question, ideally with a citation so they can trust it.

Today we close the loop. By end of day, DocuMind will:
1. Take a real PDF, break it into chunks, and store it (the **ingestion** pipeline)
2. Answer questions using retrieved chunks, **with citations** (the **generation** pipeline)
3. Say "I don't know" when it has no good information (using that threshold from yesterday)
4. Have a real way to **measure** if its answers are good (RAGAS / evaluation)

We'll also meet **LangChain** — and I'll be honest with you about when it helps and when it just adds confusion.

Let me give you the map first, because RAG has a lot of moving parts and it's easy to get lost.

---

## Part 1: What RAG actually is (the whole picture)

RAG = **R**etrieval **A**ugmented **G**eneration. Three words, three steps:

```
RETRIEVAL          AUGMENTED              GENERATION
(find chunks)  →   (stuff them in     →  (LLM writes the
                    the prompt)            answer)
```

Remember from Day 3: LLMs have no memory and make stuff up when they don't know. RAG fixes both problems with one trick — **before asking the LLM the question, we find the relevant facts and paste them into the prompt.** The LLM then answers *from those facts* instead of from its (possibly wrong, possibly outdated) memory.

Here's the analogy that makes it stick:

> **Closed-book exam vs open-book exam.**
> A raw LLM is a student taking a closed-book exam — they answer from memory, and sometimes they confidently write nonsense.
> RAG turns it into an open-book exam — we hand the student the exact textbook pages they need, then ask the question. Now they answer from the page in front of them.

The full DocuMind pipeline now looks like this:

```
                    INGESTION (happens once, ahead of time)
   ┌──────────────────────────────────────────────────────────┐
   │  PDF → extract text → split into chunks → embed → store    │
   └──────────────────────────────────────────────────────────┘

                    QUERY TIME (happens per question)
   ┌──────────────────────────────────────────────────────────┐
   │  question                                                  │
   │     │                                                      │
   │     ▼                                                      │
   │  retrieve (hybrid search + rerank)  ← Day 3, done          │
   │     │                                                      │
   │     ▼                                                      │
   │  build prompt with chunks  ← "augmented"                   │
   │     │                                                      │
   │     ▼                                                      │
   │  LLM generates answer with citations  ← "generation"       │
   └──────────────────────────────────────────────────────────┘
```

We built the retrieval half yesterday. Today: ingestion (the top box) and generation (the bottom two boxes).

---

## Part 2: Document Ingestion — and why chunking is an art

### Why we can't just dump a whole PDF in

You might think: "Why chunk at all? Just embed the whole document." Three reasons:

1. **Embedding models have input limits** (~8K tokens). A 100-page PDF won't fit.
2. **Precision.** If you embed a whole 50-page manual as one vector, searching for "how to reset password" returns the *whole manual* — not the one paragraph you need. Small chunks = precise retrieval.
3. **Context cost.** You only want to paste the *relevant* paragraphs into the LLM prompt, not the entire document (remember: every token costs money).

So we split documents into **chunks** — small, self-contained pieces. But *how* you split matters enormously. This is where most RAG systems quietly fail.

### The chunking problem, illustrated

Imagine this text in a manual:

```
To reset your password, follow these steps. First, go to the
settings page. Then click "Account." Finally, click the reset
button and check your email.
```

**Bad chunking (fixed 50-character cuts):**
```
Chunk 1: "To reset your password, follow these steps. First,"
Chunk 2: "go to the settings page. Then click "Account." Fina"
Chunk 3: "lly, click the reset button and check your email."
```

See the problem? The instructions got chopped mid-sentence. If a user searches "how to reset password," they might get Chunk 1 — which says "follow these steps" but doesn't contain the actual steps! The answer is split across three chunks and none of them is complete.

**Good chunking (respect boundaries + overlap):**
```
Chunk 1: "To reset your password, follow these steps. First, go
          to the settings page. Then click "Account." Finally,
          click the reset button and check your email."
```

Self-contained. Complete instructions. Searchable as one unit.

### The chunking strategies you should know

There's a ladder of sophistication. Let me walk you up it:

**Level 1 — Fixed-size chunking**
Split every N characters/tokens. Simple, fast, dumb. Cuts mid-sentence. Only use as a last resort.

**Level 2 — Recursive character chunking (the workhorse)**
Try to split on paragraph breaks first (`\n\n`). If a chunk is still too big, split on sentences (`. `). Still too big? Split on words. This respects natural boundaries. **This is what 80% of production systems use.** It's the sensible default.

**Level 3 — Semantic chunking**
Use embeddings to detect *topic shifts* — when consecutive sentences stop being about the same thing, that's a chunk boundary. More accurate, more expensive (you embed while chunking). Worth it for high-value documents.

**Level 4 — Structure-aware chunking**
Use the document's own structure — markdown headers, PDF sections, HTML tags — as boundaries. Best for well-structured docs (technical manuals, legal contracts). A chunk = a section.

### The two parameters that matter: size and overlap

**Chunk size** — how big each piece is. Common range: 256–1024 tokens.
- Smaller chunks = more precise retrieval, but might lose context
- Larger chunks = more context, but noisier retrieval and higher cost

**Chunk overlap** — how much consecutive chunks share. Common: 10–20% of chunk size.

Why overlap? So you don't lose information that sits on a boundary:

```
No overlap:
  Chunk 1: "...the system uses OAuth for authentication."
  Chunk 2: "This token expires after 30 minutes..."
            ↑ "This token" — WHICH token? Context lost!

With overlap:
  Chunk 1: "...the system uses OAuth for authentication."
  Chunk 2: "...uses OAuth for authentication. This token
            expires after 30 minutes..."
            ↑ now "This token" has its OAuth context
```

Overlap is cheap insurance against splitting related ideas apart.

### Let's build the ingestion pipeline

First, install PDF reading + a text splitter:

```bash
uv add pypdf langchain-text-splitters
```

We'll use `langchain-text-splitters` for the chunking — it's a small, focused library (not the whole LangChain framework) that implements recursive chunking really well. No need to reinvent it.

`src/documind/ingest.py`:

```python
"""
Document ingestion: PDF → text → chunks → embeddings → Qdrant.
This runs once per document, ahead of query time.
"""
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from documind.embeddings import Embedder
from documind.vectorstore import VectorStore, Chunk


def extract_pdf_text(pdf_path: str) -> list[tuple[int, str]]:
    """
    Extract text from a PDF, page by page.
    Returns a list of (page_number, text) so we can cite page numbers later.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            pages.append((i, text))
    return pages


def chunk_pages(
    pages: list[tuple[int, str]],
    source_name: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """
    Split page text into overlapping chunks using recursive splitting.
    Keeps track of which page each chunk came from (for citations).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try these separators in order — paragraph, line, sentence, word
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    chunk_idx = 0
    for page_num, page_text in pages:
        pieces = splitter.split_text(page_text)
        for piece in pieces:
            chunks.append(Chunk(
                text=piece,
                source=source_name,
                chunk_index=chunk_idx,
                extra={"page": page_num},  # ← page number for citation
            ))
            chunk_idx += 1
    return chunks


async def ingest_pdf(
    pdf_path: str,
    embedder: Embedder,
    store: VectorStore,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> int:
    """
    Full ingestion pipeline for one PDF.
    Returns the number of chunks stored.
    """
    source_name = Path(pdf_path).name

    # 1. Extract
    pages = extract_pdf_text(pdf_path)
    print(f"  Extracted {len(pages)} pages from {source_name}")

    # 2. Chunk
    chunks = chunk_pages(pages, source_name, chunk_size, chunk_overlap)
    print(f"  Split into {len(chunks)} chunks")

    # 3. Embed (in batches — embedding 1000 chunks at once may hit limits)
    BATCH = 100
    all_embeddings = []
    for i in range(0, len(chunks), BATCH):
        batch_texts = [c.text for c in chunks[i:i + BATCH]]
        batch_emb = await embedder.embed_many(batch_texts)
        all_embeddings.extend(batch_emb)
        print(f"  Embedded {min(i + BATCH, len(chunks))}/{len(chunks)} chunks")

    # 4. Store
    await store.add(chunks, all_embeddings)
    print(f"  Stored {len(chunks)} chunks in Qdrant")

    return len(chunks)
```

Notice the production-shaped decisions:

- **We track the page number** in `extra={"page": page_num}`. This flows all the way through so the final answer can say "according to page 7..." That's what makes citations possible.
- **We batch embeddings** (`BATCH = 100`). Embedding 5000 chunks in one API call would fail or be slow. Batching is how real ingestion pipelines work.
- **Recursive splitter with ordered separators** — paragraph first, then line, then sentence, then word. This is the Level 2 strategy.

Now an ingest script. Grab any PDF you have — a research paper, a manual, anything. Or download one. `scripts/04_ingest.py`:

```python
"""Ingest a PDF into DocuMind."""
import asyncio
import sys
from documind.embeddings import Embedder
from documind.vectorstore import VectorStore
from documind.ingest import ingest_pdf


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/04_ingest.py <path_to_pdf>")
        return

    pdf_path = sys.argv[1]

    embedder = Embedder()
    store = VectorStore(dim=embedder.dim)
    await store.setup()

    print(f"Ingesting {pdf_path}...")
    n = await ingest_pdf(pdf_path, embedder, store)
    print(f"\nDone — {n} chunks now searchable.")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
uv run python scripts/04_ingest.py /path/to/some/document.pdf
```

---

## Part 3: Generation — turning chunks into cited answers

Now the payoff. We have retrieval (Day 3) and ingestion (just now). Let's generate answers.

The core idea is a carefully designed **prompt** that:
1. Gives the LLM the retrieved chunks (with their source + page)
2. Tells it to answer ONLY from those chunks
3. Tells it to cite which chunk it used
4. Tells it to say "I don't know" if the chunks don't contain the answer

This prompt is the single most important piece of any RAG system. Let me show it, then explain every line.

`src/documind/rag.py`:

```python
"""
The RAG answer pipeline: retrieve → build prompt → generate cited answer.
This is where DocuMind actually answers questions.
"""
from dataclasses import dataclass
from groq import AsyncGroq

from documind.config import settings
from documind.retriever import Retriever
from documind.vectorstore import SearchResult


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]   # [{source, page, text, score}, ...]
    used_context: bool    # did we find relevant chunks at all?


RAG_SYSTEM_PROMPT = """You are DocuMind, a precise technical documentation assistant.

Rules you must follow:
1. Answer the user's question using ONLY the provided context below.
2. If the context does not contain the answer, say "I don't have information \
about that in the provided documents." Do not use outside knowledge.
3. Cite your sources inline using [1], [2] etc. matching the numbered context.
4. Be concise and accurate. Do not pad the answer.
5. If you are uncertain, say so."""


def _build_context(results: list[SearchResult]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, r in enumerate(results, start=1):
        page = r.chunk.extra.get("page", "?") if r.chunk.extra else "?"
        blocks.append(
            f"[{i}] (source: {r.chunk.source}, page: {page})\n{r.chunk.text}"
        )
    return "\n\n".join(blocks)


class RAGPipeline:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm = AsyncGroq(api_key=settings.groq_api_key)

    async def answer(
        self,
        question: str,
        k_final: int = 5,
        source_filter: str | None = None,
    ) -> RAGAnswer:
        # 1. Retrieve (hybrid search + rerank + threshold from Day 3)
        results = await self.retriever.retrieve(
            query=question,
            k_final=k_final,
            source_filter=source_filter,
        )

        # 2. No good chunks? Say so honestly — don't make things up.
        if not results:
            return RAGAnswer(
                answer="I don't have information about that in the provided documents.",
                sources=[],
                used_context=False,
            )

        # 3. Build the augmented prompt
        context = _build_context(results)
        user_message = f"""Context:

{context}

---

Question: {question}

Answer using only the context above, citing sources with [n]."""

        # 4. Generate
        resp = await self.llm.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,   # low — we want faithful, not creative
            max_tokens=1024,
        )
        answer_text = resp.choices[0].message.content or ""

        # 5. Package the sources so the UI can show them
        sources = [
            {
                "n": i + 1,
                "source": r.chunk.source,
                "page": r.chunk.extra.get("page", "?") if r.chunk.extra else "?",
                "score": round(r.score, 3),
                "text": r.chunk.text[:200],
            }
            for i, r in enumerate(results)
        ]

        return RAGAnswer(
            answer=answer_text,
            sources=sources,
            used_context=True,
        )
```

Let me walk through the design choices, because each one is there for a reason:

**The system prompt is everything.** Look at rule 2: *"If the context does not contain the answer, say I don't have information... Do not use outside knowledge."* Without this, the LLM falls back on its training data and hallucinates. This one rule is the difference between "trustworthy RAG" and "confident liar."

**Numbered context blocks `[1] [2]`.** We number each chunk and tell the LLM to cite with `[n]`. Now the answer comes back like "To reset your password, go to Settings [2]." The user can click `[2]` and see exactly which page that came from. **Citations are what make RAG trustworthy** — they let the user verify.

**`temperature=0.1`.** Remember Day 3 — low temperature for factual tasks. We don't want creative answers, we want faithful ones.

**The empty-results path.** Because of the threshold we added yesterday, `retrieve()` returns `[]` when nothing is relevant. Here we catch that and say "I don't know" *before* even calling the LLM. Saves money and prevents hallucination.

**We return `sources` separately.** The answer text has `[1] [2]` markers; the `sources` list maps those numbers to actual source/page/text. The frontend renders these as clickable citations.

### Wire it into the API

Add an `/ask` endpoint. `src/documind/api/routes/ask.py`:

```python
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    source: str | None = None  # optional: restrict to one document


@router.post("/ask")
async def ask(req: AskRequest, request: Request):
    rag = request.app.state.rag
    result = await rag.answer(req.question, source_filter=req.source)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "used_context": result.used_context,
    }
```

And wire the RAG pipeline into the app's `lifespan` in `app.py`:

```python
from documind.retriever import Retriever
from documind.rag import RAGPipeline
from documind.api.routes import ask  # new

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq = AsyncGroq(api_key=settings.groq_api_key)

    retriever = Retriever()
    await retriever.setup()
    app.state.rag = RAGPipeline(retriever)   # ← new

    yield
    await app.state.groq.close()

# add the new router
app.include_router(ask.router, prefix="/v1")
```

Now test the whole thing end to end:

```bash
# 1. Ingest a PDF
uv run python scripts/04_ingest.py /path/to/document.pdf

# 2. Start the server
uv run uvicorn documind.api.app:app --reload --port 8000

# 3. Ask a question about the PDF
curl -X POST http://localhost:8000/v1/ask \
  -H "content-type: application/json" \
  -d '{"question": "What is the main topic of this document?"}'
```

You should get back a JSON answer with inline `[1] [2]` citations and a `sources` list showing which pages they came from. **That's a complete RAG system.** Ingest, retrieve, rerank, generate, cite. You built it.

---

## Part 4: LangChain & LCEL — the honest take

The curriculum lists "LangChain + LCEL basics." Let me give you the truth instead of the marketing.

### What LangChain is

LangChain is a framework that provides pre-built components for LLM apps — chains, prompt templates, retrievers, output parsers, document loaders — and a way to wire them together. **LCEL** (LangChain Expression Language) is its syntax for chaining components with the `|` pipe operator:

```python
# LCEL — the | operator chains steps together
chain = prompt | llm | output_parser
result = chain.invoke({"question": "What is RAG?"})
```

It reads like a Unix pipe: take the prompt, pipe it to the LLM, pipe that to the parser.

### Here's how our RAG pipeline would look in LangChain

```python
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template("""
Answer using only this context:
{context}

Question: {question}
""")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

# The chain: retrieve context, fill prompt, call LLM, parse output
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is RAG?")
```

That's genuinely elegant for simple cases. Four lines and you have a RAG chain.

### My honest opinion (and why we built it by hand first)

**The good:**
- Great for prototyping — get a RAG demo running in 10 minutes
- Huge ecosystem of document loaders (PDF, Notion, Slack, web pages) — genuinely useful, saves you writing parsers
- The text splitters (which we already used) are excellent
- Standard interface across providers

**The bad:**
- **Magic hides understanding.** If we'd started with that 4-line chain, you'd have a working RAG system but no idea *why* it works. You wouldn't understand chunking, reranking, the prompt design, the citation mechanism, or the "I don't know" path. When it breaks (and it will), you'd be lost.
- **Debugging is painful.** When a LangChain chain produces a bad answer, figuring out *which* step went wrong means digging through layers of abstraction.
- **It changes constantly.** LangChain's API has churned heavily across versions — chains you wrote last year may not work today.
- **You lose control.** Our hand-built version lets us insert the threshold check, the page-number tracking, the custom citation format. Bending LangChain to do non-standard things often fights the framework.

**My actual recommendation for you:**
- **Learn the patterns by hand** (which we did) so you understand what's happening
- **Use LangChain's narrow utilities** where they save real work — document loaders, text splitters
- **For the orchestration core, prefer either raw SDK calls (what we did) or LangGraph** (the one LangChain-family library I trust for production — we'll use it on Day 6)
- **Don't tattoo any framework on yourself.** The understanding is permanent; the framework is disposable.

So: I want you to *be able* to read and write LCEL — you'll see it everywhere, and some teams standardize on it. But I built DocuMind by hand on purpose, because the goal is for you to *understand RAG*, not to *understand LangChain*.

### Quick exercise to cement it
Install `langchain-groq` and rewrite just the generation step as an LCEL chain. Compare it to our hand-built `RAGPipeline.answer`. Notice what's cleaner (the chaining) and what's lost (the threshold, citations, page tracking). That comparison *is* the lesson.

---

## Part 5: Evaluation — how do you know if your RAG is any good?

This is the part almost everyone skips, and it's the part that separates toys from production systems. Here's the uncomfortable truth:

> **You cannot improve what you cannot measure.** If you change your chunk size from 800 to 500, did your RAG get better or worse? Without evaluation, you're just guessing.

### Why RAG evaluation is hard

You can't use `assertEqual`. There's no single "correct" answer string — many phrasings are equally good. So we need *fuzzy* evaluation that judges quality. The standard approach is **LLM-as-judge** plus a set of RAG-specific metrics.

### The four RAGAS metrics (learn these — they're the industry standard)

RAGAS is a library that measures RAG quality on four dimensions. Even if you never use the library, **the four concepts are what you must understand:**

```
                    Question
                   /         \
                  /           \
            RETRIEVAL      GENERATION
            (did we find    (did we answer
             good chunks?)   well?)
```

**1. Context Precision** — *Of the chunks we retrieved, how many were actually relevant?*
Measures retrieval quality. Low precision = your retriever is pulling in junk. (This is exactly what reranking improves.)

**2. Context Recall** — *Of all the relevant info that exists, how much did we retrieve?*
Measures retrieval completeness. Low recall = you're missing chunks that contained the answer. (Often a chunking problem.)

**3. Faithfulness** — *Does the answer stick to the retrieved context, or did the LLM make things up?*
Measures hallucination. Low faithfulness = the LLM is using outside knowledge or inventing facts. This is the scariest failure — confident wrong answers.

**4. Answer Relevancy** — *Does the answer actually address the question?*
Measures whether the answer is on-topic. Low relevancy = the LLM wandered off or gave a vague non-answer.

A great mental model:
- Precision + Recall judge **the retriever** (Day 3's job)
- Faithfulness + Relevancy judge **the generator** (today's prompt)

When DocuMind gives a bad answer, these four metrics tell you *where* it broke. Bad faithfulness? Fix the prompt. Bad recall? Fix the chunking. This is how you debug RAG systematically instead of randomly fiddling.

### Let's build a simple evaluator by hand

Before reaching for RAGAS the library, let's build a faithfulness checker ourselves — because understanding it manually means you'll actually understand what RAGAS reports. We use an LLM as the judge.

`src/documind/evaluate.py`:

```python
"""
Simple RAG evaluation using LLM-as-judge.
Builds the intuition behind RAGAS metrics by computing two of them by hand.
"""
import json
from groq import AsyncGroq
from documind.config import settings


class RAGEvaluator:
    def __init__(self):
        self.llm = AsyncGroq(api_key=settings.groq_api_key)
        # Use a strong model as the judge — judging needs good reasoning
        self.judge_model = settings.groq_model

    async def faithfulness(self, answer: str, context: str) -> dict:
        """
        Does the answer stick to the context, or did it make things up?
        Returns a score 0-1 and the reasoning.
        """
        prompt = f"""You are evaluating whether an AI answer is faithful to its source context.

CONTEXT:
{context}

ANSWER:
{answer}

Check every claim in the ANSWER. Is each claim supported by the CONTEXT?

Respond ONLY with JSON:
{{"score": <0.0 to 1.0>, "unsupported_claims": ["..."], "reasoning": "..."}}

score 1.0 = every claim supported by context
score 0.0 = answer contains claims not in context (hallucination)"""

        resp = await self.llm.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)

    async def answer_relevancy(self, question: str, answer: str) -> dict:
        """Does the answer actually address the question?"""
        prompt = f"""You are evaluating whether an answer addresses a question.

QUESTION: {question}

ANSWER: {answer}

Respond ONLY with JSON:
{{"score": <0.0 to 1.0>, "reasoning": "..."}}

score 1.0 = directly and fully answers the question
score 0.0 = does not address the question at all"""

        resp = await self.llm.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content)
```

And an eval script that runs your RAG against a set of test questions. `scripts/04_eval.py`:

```python
"""Evaluate DocuMind's RAG quality on a set of test questions."""
import asyncio
from documind.retriever import Retriever
from documind.rag import RAGPipeline, _build_context
from documind.evaluate import RAGEvaluator

# Your "golden" test set — questions you know the docs can answer.
# In production this grows to 50-200 questions covering edge cases.
TEST_QUESTIONS = [
    "What is the main topic of the document?",
    "What are the key steps described?",
    # add questions specific to YOUR ingested PDF
]


async def main():
    retriever = Retriever()
    await retriever.setup()
    rag = RAGPipeline(retriever)
    evaluator = RAGEvaluator()

    faithfulness_scores = []
    relevancy_scores = []

    for q in TEST_QUESTIONS:
        result = await rag.answer(q)

        if not result.used_context:
            print(f"Q: {q}\n  → No context found (skipping)\n")
            continue

        # Rebuild the context string that was used (for the judge)
        context = "\n\n".join(s["text"] for s in result.sources)

        faith = await evaluator.faithfulness(result.answer, context)
        rel = await evaluator.answer_relevancy(q, result.answer)

        faithfulness_scores.append(faith["score"])
        relevancy_scores.append(rel["score"])

        print(f"Q: {q}")
        print(f"  Answer: {result.answer[:100]}...")
        print(f"  Faithfulness: {faith['score']:.2f}")
        print(f"  Relevancy:    {rel['score']:.2f}")
        if faith["score"] < 1.0:
            print(f"  ⚠️  Unsupported: {faith.get('unsupported_claims')}")
        print()

    if faithfulness_scores:
        print("─" * 40)
        print(f"Avg Faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.2f}")
        print(f"Avg Relevancy:    {sum(relevancy_scores)/len(relevancy_scores):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
```

Run it after ingesting a PDF and you'll get a quality scorecard. Now when you tweak chunk size or the prompt, **re-run this and see if the numbers go up or down.** That's the entire game of RAG improvement.

### About RAGAS the library

RAGAS (the actual `ragas` package) automates all four metrics and adds more. It's worth using once you understand the concepts. A caveat I'll repeat from Day 3: RAGAS's API has changed significantly across versions, so check its current docs rather than trusting any code snippet (including one I'd write from memory). The *concepts* — precision, recall, faithfulness, relevancy — are stable and what you actually need to carry forward. The library is a convenience wrapper around exactly the LLM-as-judge idea you just built by hand.

---

## Day 4 exercises

**Exercise 4.1 — Ingest a real document**
Find a PDF (a paper, a manual, anything 5+ pages). Ingest it. Then ask it 3 questions through the `/ask` endpoint. Paste me one answer with its citations.

**Exercise 4.2 — Chunk size experiment**
Re-ingest the same PDF three times with different chunk sizes: 300, 800, 1500 (delete the collection between runs). Ask the same question each time. How do the answers change? Smaller chunks should give more precise but possibly incomplete answers; bigger chunks more context but noisier retrieval. Report what you observe.

**Exercise 4.3 — Break faithfulness on purpose**
Ask DocuMind a question that's *not* covered in your PDF (e.g., if you ingested a Python manual, ask "what's the capital of France?"). Confirm it says "I don't have information about that." Then temporarily *remove* rule 2 from the system prompt and ask again. Watch it hallucinate from outside knowledge. Put the rule back. This shows you exactly what that one prompt line is protecting against.

**Exercise 4.4 — Build your golden test set**
Write 5 questions you *know* your PDF can answer, plus 2 that it *can't*. Run `04_eval.py`. The 5 answerable ones should score high on faithfulness and relevancy; the 2 unanswerable ones should return "I don't know" (used_context=False). This is your first real eval set — the thing you'll re-run every time you change something.

**Exercise 4.5 — LCEL comparison (optional)**
Rewrite the generation step using LangChain's LCEL (`prompt | llm | parser`). Compare line count and readability to our hand-built version. Note what you'd lose (citations, threshold, page tracking). Tell me which you'd choose for a real project and why.

**Exercise 4.6 — Explain it back**
In your own words:
1. Why do we chunk documents instead of embedding them whole?
2. What does chunk overlap protect against?
3. What's the difference between faithfulness and answer relevancy?
4. Why is the "answer only from context" rule the most important line in the prompt?

---

## What you should walk away with

- **RAG = open-book exam for an LLM.** Find the right pages, paste them in, then ask.
- **Chunking is an art** — recursive splitting with overlap is the sensible default; size and overlap are the dials.
- **The system prompt is the soul of RAG** — "answer only from context" prevents hallucination; numbered citations create trust.
- **The "I don't know" path** (from your Day 3 threshold) is a feature, not a failure.
- **You can't improve what you can't measure** — the four RAGAS metrics (precision, recall, faithfulness, relevancy) tell you *where* RAG breaks.
- **LangChain is a tool, not a religion** — use its loaders and splitters, understand LCEL, but don't let it hide how things work.

---

DocuMind is now a real, working RAG system: it ingests PDFs, retrieves with hybrid search and reranking, generates cited answers, refuses to hallucinate, and can measure its own quality. That's genuinely more than many shipped products do.

Say **"next"** for **Day 5: Agentic AI Foundation** — where DocuMind stops just *answering* and starts *acting*. We'll build a ReAct agent from scratch, give it tools (like web search and its own RAG retriever), add a reasoning loop, and meet the reliability problems that make agents hard in production. This is where it gets exciting.

And if you want, I can bundle today into a Day 4 notes `.md` like I did for Day 3 — just say so.
