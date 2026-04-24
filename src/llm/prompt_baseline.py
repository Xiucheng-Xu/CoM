"""Baseline Model Prompts"""

# ==================== Baseline Prompts for LongMemEval====================

BASELINE_SYSTEM_PROMPT = """You are a helpful expert assistant answering questions from the user based on the provided context."""

BASELINE_PROMPT = """
Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.

Context: {context}

Question: {question}
"""

# Mem0 Answer Prompt - 用于基于检索的记忆回答问题（单用户版本）
# 改编自: baseline/mem0/evaluation/prompts.py 的 ANSWER_PROMPT
MEM0_ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from past conversations. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), 
   calculate the actual date based on the memory timestamp. For example, if a memory from 
   4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, 
   convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
   timestamp.
7. Focus only on the content of the memories
8. The answer should be concise and direct

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation (e.g., converting relative time references), show your work
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references

Context:
{context}

Question: {question}

Please provide your answer in JSON format with an "answer" field."""

# ==================== Mem0 Prompts ====================

# Mem0 Custom Instructions - 用于记忆提取
# 源自: baseline/mem0/evaluation/src/memzero/add.py
# 为LongMemEval单用户场景做了调整
MEM0_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred
   - Context from the conversation

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones
   - User preferences and interests
   - Key information from conversations

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements
   - Include contextual information

4. Extract memories from both user and assistant messages when they contain important information

5. Format each memory as a clear statement or paragraph that captures the person's experience, challenges, and aspirations
"""

# ==================== LoComo Prompts ====================

# System Prompt（角色定义和指令）
SYSTEM_PROMPT_LOCOMO = """You are a **Memory-Based Question Answering Agent**. Answer questions based on the provided memories.

<Instructions>
1. **For time-related questions(with word like "when")**:
   - Step 1: Identify the Timestamp of the memory. The Timestamp is the date when the speaker said something.
   - Step 2: Identify any relative time reference (last week, yesterday, X ago, last Friday) in the text of the memory. The relative time reference indicates when the event occurred relative to the Timestamp of the memory.
   - Step 3: Calculate the **EXACT ABSOLUTE** date when the event happened by combining the Timestamp of the memory and the relative time reference in the text of the memory.
   - Example: [Timestamp: 4 May 2022] + "last year" → Answer: "last year before 4 May 2022(or 4 May 2021)" 
   - Example: [Timestamp: 17 July 2023] + "last Friday" → Answer: "last Friday before 17 July 2023"
2. **Extract specific details**: Use exact names, dates, places from memories. Never give vague answers like "her home country" - find and state the exact name (e.g., "Sweden").
3. **Be comprehensive for list questions**: If asked "What activities does X do?", list ALL activities found across all memories.
4. **Contradictions**: If memories conflict, use the most recent one (latest Timestamp).
</Instructions>"""

# User Prompt（memories 和 question）
USER_PROMPT_LOCOMO = """<Memories>
{memories}
</Memories>

<Question>{question}</Question>

Answer:"""