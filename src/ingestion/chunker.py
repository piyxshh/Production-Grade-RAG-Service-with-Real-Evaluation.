# [CORE LEARNING — write this yourself]
# Chunking logic: take a document's text and split it into chunks.
#
# Start with fixed-size chunking with overlap.
# Then experiment: what chunk size produces the most meaningful retrieval units?
#
# Questions to answer before you start:
# 1. What does a chunk promise to the retrieval system?
# 2. What happens when a key fact spans a chunk boundary?
# 3. How does chunk size affect embedding quality?
