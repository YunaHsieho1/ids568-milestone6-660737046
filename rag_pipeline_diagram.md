# Part 1 - RAG Pipeline Diagram

```
                    +------------------+
                    | Source docs (.txt) |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |   Text Loader    |
                    | keep source name |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |     Chunker      |
                    | size + overlap   |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |    Embedder      |
                    | sentence-transformers
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Vector Index    |
                    |  (FAISS IndexFlatIP)
                    +--------+---------+
                             ^
                             |
              +------------+-------------+
              |                          |
              |  query embedding         |
              v                          |
     +----------------+        +--------+---------+
     | User question  |------->|    Retriever     |
     +----------------+        | top-k + scores   |
                                 +--------+---------+
                                          |
                                          v
                                 +------------------+
                                 | Prompt Builder   |
                                 | Context + rules  |
                                 +--------+---------+
                                          |
                                          v
                                 +------------------+
                                 |  LLM Generator   |
                                 | (Ollama 7B model)|
                                 +--------+---------+
                                          |
                                          v
                                 +------------------+
                                 | Grounded answer  |
                                 +------------------+
```

Decision points and transformations:

1. **Chunking policy:** fixed-size windows with overlap to avoid sentence boundary loss.  
2. **Similarity policy:** L2-normalized vectors with inner product (cosine-equivalent ranking).  
3. **Grounding policy:** generator prompt requires using retrieved context and returning source files.
