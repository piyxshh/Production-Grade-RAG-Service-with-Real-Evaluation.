# [CORE LEARNING — write this yourself]
# Dense retrieval: given a query embedding, return the top-k most similar chunks
# from Postgres using pgvector.
#
# Use the <=> operator for cosine distance (or <#> for inner product — pick one, know why).
#
# Questions to answer before you start:
# 1. What does cosine similarity measure? What does inner product measure?
# 2. What is ANN (Approximate Nearest Neighbor) vs exact search?
# 3. What does pgvector's IVFFlat index do, and when should you use it?
