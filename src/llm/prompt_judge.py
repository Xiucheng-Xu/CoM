"""LLM as Judge Evaluation Prompts"""

# ==================== LongMemEval LLM as Judge Evaluation Prompts ====================

# Basic evaluation prompt (for single-session-user, single-session-assistant, multi-session)
EVAL_BASIC_PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. 

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.

/no_think"""

# Temporal reasoning evaluation prompt
EVAL_TEMPORAL_REASONING_PROMPT = """I will give you a question, a correct answer, and a response from a model. 
Please answer yes if the response contains the correct answer. Otherwise, answer no. 
If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. 
If the response only contains a subset of the information required by the answer, answer no. 
In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.
If the model response uses a relative time expression and it evaluates to approximately the same date as the correct answer, the response is still correct (e.g., "the week before June 3" vs "late May" are considered equivalent). 

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.

/no_think"""

# Knowledge update evaluation prompt
EVAL_KNOWLEDGE_UPDATE_PROMPT = """I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Question: {}

Correct Answer: {}

Model Response: {}

Is the model response correct? Answer yes or no only.

/no_think"""

# Preference evaluation prompt
EVAL_PREFERENCE_PROMPT = """I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.

Question: {}

Rubric: {}

Model Response: {}

Is the model response correct? Answer yes or no only.

/no_think"""

# Abstention/unanswerable evaluation prompt
EVAL_ABSTENTION_PROMPT = """I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.

Question: {}

Explanation: {}

Model Response: {}

Does the model correctly identify the question as unanswerable? Answer yes or no only.

/no_think"""


# ==================== Locomo LLM as Judge Evaluation Prompts ====================
LOCOMO_JUDGE_PROMPT = """You are an expert judge evaluating whether a model's prediction correctly answers a question compared to the reference answer.

Question: {}

Reference Answer: {}

Model Prediction: {}

Your task is to determine if the model's prediction is semantically equivalent to the reference answer. Consider the following:
1. The prediction may be phrased differently but convey the same meaning
2. Minor differences in wording are acceptable if the core information matches
3. For dates, consider different formats as equivalent (e.g., "7 May 2023" vs "May 7, 2023")
4. For numbers, consider "2022" vs "Last year" as potentially equivalent depending on context
5. For descriptive answers, check if the key information is present

Respond with ONLY ONE WORD:
- "CORRECT" if the prediction matches the reference answer
- "INCORRECT" if the prediction does not match the reference answer

Your response:

/no_think"""