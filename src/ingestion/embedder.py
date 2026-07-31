# [CORE LEARNING — write this yourself]
# Embedder: take a list of text strings, return a list of embedding vectors.
#
# Critical requirements:
# - Must be async (use asyncio.gather for concurrent calls, NOT sequential)
# - Must handle rate limits with exponential backoff (use tenacity)
# - Must support batching (the API has a per-request token limit)
#
# Questions to answer before you start:
# 1. Why does async matter here specifically?
# 2. What happens if you call the API sequentially for 10,000 chunks?
# 3. What is the difference between an embedding and a token?
