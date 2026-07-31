# [CORE LEARNING — write this yourself]
# Cross-encoder reranker: given a query and a list of candidate chunks,
# return the top-n chunks reranked by a cross-encoder model.
#
# Use sentence-transformers with a cross-encoder model:
#   cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, good quality)
#
# Questions to answer before you start:
# 1. What is the architectural difference between a bi-encoder (vector search)
#    and a cross-encoder?
# 2. Why can't you use a cross-encoder for initial retrieval over your entire corpus?
# 3. What does a cross-encoder score represent?
