# [CORE LEARNING — write this yourself]
# Prompt assembly: take a query + list of reranked chunks and build a grounded prompt.
#
# Requirements:
# - Include source metadata (chunk ID, document title) so the LLM can cite sources
# - Make it explicit in the prompt that the LLM should NOT answer outside the context
# - Experiment: what prompt format produces more faithful answers in RAGAS evaluation?
