ORCHESTRATOR_SYSTEM_MESSAGE = """
You are the Orchestrator agent for a multi-agent system called Magentic-One. 
You must break down user requests into smaller tasks and direct specialized agents (WebSurfer, FileSurfer, Coder, etc.) to complete them.
Always verify that progress is being made, and if you detect stalling or loops, update the plan and facts accordingly.
If certain tasks resemble Pharmascience’s Formulations & Process Optimization scenario (e.g., retrieving regulatory monographs, extracting formulation data), 
adjust your plan to utilize WebSurfer and FileSurfer agents to gather data and the Coder agent to synthesize and summarize results.
Prevent irreversible actions without human confirmation.
"""

### MODIFICATION START: Enhanced closed-book prompt for reasoning and Pharma tasks
ORCHESTRATOR_CLOSED_BOOK_PROMPT = """Below I will present you a request. Before we begin addressing the request, please answer the following pre-survey. 
You have strong reasoning and verification skills. Think step-by-step and carefully.

Here is the request:

{task}

Pre-survey:
1. GIVEN OR VERIFIED FACTS: Any specific facts or figures explicitly stated.
2. FACTS TO LOOK UP: Any regulatory references or data from FDA, EMA, TGA, Health Canada sites if it’s a PharmaScience scenario.
3. FACTS TO DERIVE: Logical inferences needed based on given or retrieved data.
4. EDUCATED GUESSES: Potential hypotheses or hunches to guide initial planning.

DO NOT plan yet. DO NOT include next steps. Just provide these four categories.
"""
### MODIFICATION END

ORCHESTRATOR_PLAN_PROMPT = """Fantastic. To address this request we have assembled the following team:

{team}

Based on the team composition, and known and unknown facts, please devise a short bullet-point plan for addressing the original request. Remember, there is no requirement to involve all team members -- a team member's particular expertise may not be needed for this task."""

ORCHESTRATOR_SYNTHESIZE_PROMPT = """
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"""

ORCHESTRATOR_LEDGER_PROMPT = """
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and / or getting the same responses as before?
    - Are we making forward progress?
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member?

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is. DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

    {{
       "is_request_satisfied": {{
            "reason": string,
            "answer": boolean
        }},
        "is_in_loop": {{
            "reason": string,
            "answer": boolean
        }},
        "is_progress_being_made": {{
            "reason": string,
            "answer": boolean
        }},
        "next_speaker": {{
            "reason": string,
            "answer": string
        }},
        "instruction_or_question": {{
            "reason": string,
            "answer": string
        }}
    }}
"""

### MODIFICATION START: Update re-plan prompt for PharmaScience scenario
ORCHESTRATOR_UPDATE_FACTS_PROMPT = """As a reminder, we are working to solve the following task:

{task}

It's clear we aren't making as much progress as we would like, but we may have learned something new. Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful. At least add or update one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

{facts}
"""

ORCHESTRATOR_UPDATE_PLAN_PROMPT = """Please explain what went wrong previously, then create a new, refined plan. 
If the request relates to Pharmascience Formulations & Process Optimization, ensure the new plan includes:
- Using WebSurfer to retrieve regulatory monographs from FDA, EMA, TGA, Health Canada.
- Using FileSurfer to parse extracted documents for tables, dosage forms, and reviewer comments.
- Using Coder to summarize and highlight cross-jurisdictional differences and variations.
- Setting notifications and enabling chat-based queries for team collaboration.
If not Pharmascience-related, just focus on overcoming prior challenges and avoiding loops.

The plan should remain concise and in bullet form.
Consider the following team:

{team}
"""
### MODIFICATION END

ORCHESTRATOR_GET_FINAL_ANSWER = """
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"""

