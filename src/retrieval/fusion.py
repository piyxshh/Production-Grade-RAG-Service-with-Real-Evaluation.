# [CORE LEARNING — write this yourself]
# Reciprocal Rank Fusion (RRF): merge dense and sparse result lists into one.
#
# Formula: score(d) = sum over each list of 1 / (k + rank(d))
# where rank(d) is the 1-indexed position of document d in that list.
# k=60 is the standard constant — understand why before using it.
#
# Questions to answer before you start:
# 1. Why not just average the scores from both lists directly?
# 2. What does the k constant control? What happens at k=0? k=1000?
# 3. What happens to a document that appears in only one list?
