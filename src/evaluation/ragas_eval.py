# [CORE LEARNING — write this yourself]
# RAGAS evaluation harness:
# 1. Load evaluation/test_set.json
# 2. For each question, run the pipeline and collect:
#    {question, answer, contexts (list of chunks used), ground_truth}
# 3. Pass to RAGAS and get scores for:
#    - Faithfulness (is the answer supported by the retrieved context?)
#    - Answer Relevance (does the answer address the question?)
#    - Context Precision (what fraction of retrieved chunks were actually useful?)
#    - Context Recall (did you retrieve all the chunks needed to answer?)
# 4. Output scores to evaluation/results/
