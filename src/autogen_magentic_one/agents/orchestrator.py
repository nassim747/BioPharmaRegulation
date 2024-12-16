import json
from typing import Any, Dict, List, Optional

from autogen_core import AgentProxy, CancellationToken, MessageContext, TopicId, default_subscription
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from ..messages import BroadcastMessage, OrchestrationEvent, ResetMessage
from .base_orchestrator import BaseOrchestrator
from .orchestrator_prompts import (
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_GET_FINAL_ANSWER,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    ORCHESTRATOR_SYSTEM_MESSAGE,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_UPDATE_PLAN_PROMPT,
)

@default_subscription
class LedgerOrchestrator(BaseOrchestrator):
    DEFAULT_SYSTEM_MESSAGES = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_MESSAGE),
    ]

    def __init__(
        self,
        agents: List[AgentProxy],
        model_client: ChatCompletionClient,
        description: str = "Ledger-based orchestrator",
        system_messages: List[SystemMessage] = DEFAULT_SYSTEM_MESSAGES,
        closed_book_prompt: str = ORCHESTRATOR_CLOSED_BOOK_PROMPT,
        plan_prompt: str = ORCHESTRATOR_PLAN_PROMPT,
        synthesize_prompt: str = ORCHESTRATOR_SYNTHESIZE_PROMPT,
        ledger_prompt: str = ORCHESTRATOR_LEDGER_PROMPT,
        update_facts_prompt: str = ORCHESTRATOR_UPDATE_FACTS_PROMPT,
        update_plan_prompt: str = ORCHESTRATOR_UPDATE_PLAN_PROMPT,
        max_rounds: int = 20,
        max_time: float = float("inf"),
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
    ) -> None:
        super().__init__(agents=agents, description=description, max_rounds=max_rounds, max_time=max_time)

        self._model_client = model_client

        # Prompt-based parameters
        self._system_messages = system_messages
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt

        self._chat_history: List[LLMMessage] = []
        self._should_replan = True
        self._max_stalls_before_replan = max_stalls_before_replan
        self._stall_counter = 0
        self._max_replans = max_replans
        self._replan_counter = 0
        self._return_final_answer = return_final_answer

        self._team_description = ""
        self._task = ""
        self._facts = ""
        self._plan = ""

    def _get_closed_book_prompt(self, task: str) -> str:
        return self._closed_book_prompt.format(task=task)

    def _get_plan_prompt(self, team: str) -> str:
        return self._plan_prompt.format(team=team)

    def _get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return self._synthesize_prompt.format(task=task, team=team, facts=facts, plan=plan)

    def _get_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return self._ledger_prompt.format(task=task, team=team, names=names)

    def _get_update_facts_prompt(self, task: str, facts: str) -> str:
        return self._update_facts_prompt.format(task=task, facts=facts)

    def _get_update_plan_prompt(self, team: str) -> str:
        return self._update_plan_prompt.format(team=team)

    async def _get_team_description(self) -> str:
        # A single string description of all agents in the team
        team_description = ""
        for agent in self._agents:
            metadata = await agent.metadata
            name = metadata["type"]
            description = metadata["description"]
            team_description += f"{name}: {description}\n"
        return team_description

    async def _get_team_names(self) -> List[str]:
        return [(await agent.metadata)["type"] for agent in self._agents]

    def _get_message_str(self, message: LLMMessage) -> str:
        if isinstance(message.content, str):
            return message.content
        else:
            result = ""
            for content in message.content:
                if isinstance(content, str):
                    result += content + "\n"
            assert len(result) > 0
        return result

    ### MODIFICATION START: Helper to detect PharmaScience-related tasks
    def _is_pharmascience_task(self) -> bool:
        keywords = ["fda", "ema", "tga", "health canada", "formulation", "monograph", "dosage"]
        return any(k in self._task.lower() for k in keywords)
    ### MODIFICATION END

    ### MODIFICATION START: Enhanced initialization for PharmaScience tasks
    async def _initialize_task(self, task: str, cancellation_token: Optional[CancellationToken] = None) -> None:
        self._task = task
        self._team_description = await self._get_team_description()

        # 1. Gather facts
        planning_conversation = [m for m in self._chat_history]

        planning_conversation.append(
            UserMessage(content=self._get_closed_book_prompt(self._task), source=self.metadata["type"])
        )
        response = await self._model_client.create(
            self._system_messages + planning_conversation, cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        planning_conversation.append(AssistantMessage(content=self._facts, source=self.metadata["type"]))

        # 2. Create a plan
        planning_conversation.append(
            UserMessage(content=self._get_plan_prompt(self._team_description), source=self.metadata["type"])
        )
        response = await self._model_client.create(
            self._system_messages + planning_conversation, cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

        # If PharmaScience scenario, add domain-specific steps
        if self._is_pharmascience_task():
            self._plan += (
                "\n- Use WebSurfer to retrieve latest regulatory monographs from FDA, EMA, TGA, Health Canada\n"
                "- Use FileSurfer to extract relevant tables, dosage forms, and reviewer comments\n"
                "- Use DomainSummarizer to produce concise comparative summaries\n"
                "- Use Coder for final checks, code execution, or data validation steps\n"
                "- Set up notifications and enable chat-based access to consolidated data\n"
            )
    ### MODIFICATION END

    async def _update_facts_and_plan(self, cancellation_token: Optional[CancellationToken] = None) -> None:
        # Called when the orchestrator decides to replan
        planning_conversation = [m for m in self._chat_history]

        # Update facts
        planning_conversation.append(
            UserMessage(content=self._get_update_facts_prompt(self._task, self._facts), source=self.metadata["type"])
        )
        response = await self._model_client.create(
            self._system_messages + planning_conversation, cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._facts = response.content
        planning_conversation.append(AssistantMessage(content=self._facts, source=self.metadata["type"]))

        # Update plan
        planning_conversation.append(
            UserMessage(content=self._get_update_plan_prompt(self._team_description), source=self.metadata["type"])
        )
        response = await self._model_client.create(
            self._system_messages + planning_conversation, cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        self._plan = response.content

        # Ensure PharmaScience-related hints are present if needed
        if self._is_pharmascience_task() and "WebSurfer" not in self._plan:
            self._plan += (
                "\n(Added) Ensure WebSurfer retrieves regulatory docs, FileSurfer parses them, and DomainSummarizer summarizes differences."
            )

    async def update_ledger(self, cancellation_token: Optional[CancellationToken] = None) -> Dict[str, Any]:
        # Updates the ledger at each turn
        max_json_retries = 10

        team_description = await self._get_team_description()
        names = await self._get_team_names()
        ledger_prompt = self._get_ledger_prompt(self._task, team_description, names)

        ledger_user_messages: List[LLMMessage] = [UserMessage(content=ledger_prompt, source=self.metadata["type"])]

        for _ in range(max_json_retries):
            ledger_response = await self._model_client.create(
                self._system_messages + self._chat_history + ledger_user_messages,
                json_output=True,
                cancellation_token=cancellation_token,
            )
            ledger_str = ledger_response.content

            try:
                assert isinstance(ledger_str, str)
                ledger_dict: Dict[str, Any] = json.loads(ledger_str)
                required_keys = [
                    "is_request_satisfied",
                    "is_in_loop",
                    "is_progress_being_made",
                    "next_speaker",
                    "instruction_or_question",
                ]
                key_error = False
                for key in required_keys:
                    if key not in ledger_dict or "answer" not in ledger_dict[key]:
                        ledger_user_messages.append(AssistantMessage(content=ledger_str, source="self"))
                        ledger_user_messages.append(
                            UserMessage(content=f"KeyError: '{key}'", source=self.metadata["type"])
                        )
                        key_error = True
                        break
                if key_error:
                    continue
                return ledger_dict
            except json.JSONDecodeError:
                pass

        raise ValueError("Failed to parse ledger information after multiple retries.")

    async def _prepare_final_answer(self, cancellation_token: Optional[CancellationToken] = None) -> str:
        # Called when the task is complete
        final_message = UserMessage(
            content=ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._task), source=self.metadata["type"]
        )
        response = await self._model_client.create(
            self._system_messages + self._chat_history + [final_message], cancellation_token=cancellation_token
        )

        assert isinstance(response.content, str)
        return response.content

    async def _handle_broadcast(self, message: BroadcastMessage, ctx: MessageContext) -> None:
        self._chat_history.append(message.content)
        await super()._handle_broadcast(message, ctx)

    ### MODIFICATION START: Enhanced agent selection logic for PharmaScience tasks
    async def _select_next_agent(
        self, message: LLMMessage, cancellation_token: Optional[CancellationToken] = None
    ) -> Optional[AgentProxy]:
        if len(self._task) == 0:
            await self._initialize_task(self._get_message_str(message), cancellation_token)
            assert len(self._task) > 0
            assert len(self._facts) > 0
            assert len(self._plan) > 0
            assert len(self._team_description) > 0

            synthesized_prompt = self._get_synthesize_prompt(self._task, self._team_description, self._facts, self._plan)
            topic_id = TopicId("default", self.id.key)
            await self.publish_message(
                BroadcastMessage(content=UserMessage(content=synthesized_prompt, source=self.metadata["type"])),
                topic_id=topic_id,
                cancellation_token=cancellation_token,
            )

            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (thought)",
                    f"Initial plan:\n{synthesized_prompt}",
                )
            )

            self._replan_counter = 0
            self._stall_counter = 0

            synthesized_message = AssistantMessage(content=synthesized_prompt, source=self.metadata["type"])
            self._chat_history.append(synthesized_message)

            return await self._select_next_agent(synthesized_message, cancellation_token)

        # Orchestrate the next step
        ledger_dict = await self.update_ledger(cancellation_token)
        self.logger.info(
            OrchestrationEvent(
                f"{self.metadata['type']} (thought)",
                f"Updated Ledger:\n{json.dumps(ledger_dict, indent=2)}",
            )
        )

        # Task complete
        if ledger_dict["is_request_satisfied"]["answer"] is True:
            self.logger.info(
                OrchestrationEvent(
                    f"{self.metadata['type']} (thought)",
                    "Request satisfied.",
                )
            )
            if self._return_final_answer:
                final_answer = await self._prepare_final_answer(cancellation_token)
                self.logger.info(
                    OrchestrationEvent(
                        f"{self.metadata['type']} (final answer)",
                        f"\n{final_answer}",
                    )
                )
            return None

        # Check for stalls
        stalled = ledger_dict["is_in_loop"]["answer"] or not ledger_dict["is_progress_being_made"]["answer"]
        if stalled:
            self._stall_counter += 1
            if self._stall_counter > self._max_stalls_before_replan:
                self._replan_counter += 1
                self._stall_counter = 0
                if self._replan_counter > self._max_replans:
                    self.logger.info(
                        OrchestrationEvent(
                            f"{self.metadata['type']} (thought)",
                            "Replan counter exceeded... Terminating.",
                        )
                    )
                    return None
                else:
                    self.logger.info(
                        OrchestrationEvent(
                            f"{self.metadata['type']} (thought)",
                            "Stalled.... Replanning...",
                        )
                    )
                    await self._update_facts_and_plan(cancellation_token)

                    # Reset everyone, then rebroadcast the new plan
                    self._chat_history = [self._chat_history[0]]
                    topic_id = TopicId("default", self.id.key)
                    await self.publish_message(ResetMessage(), topic_id=topic_id, cancellation_token=cancellation_token)

                    # Send the NEW plan
                    synthesized_prompt = self._get_synthesize_prompt(
                        self._task, self._team_description, self._facts, self._plan
                    )
                    await self.publish_message(
                        BroadcastMessage(content=UserMessage(content=synthesized_prompt, source=self.metadata["type"])),
                        topic_id=topic_id,
                        cancellation_token=cancellation_token,
                    )

                    self.logger.info(
                        OrchestrationEvent(
                            f"{self.metadata['type']} (thought)",
                            f"New plan:\n{synthesized_prompt}",
                        )
                    )

                    synthesized_message = AssistantMessage(content=synthesized_prompt, source=self.metadata["type"])
                    self._chat_history.append(synthesized_message)
                    return await self._select_next_agent(synthesized_message, cancellation_token)

        next_agent_name = ledger_dict["next_speaker"]["answer"]
        instruction = ledger_dict["instruction_or_question"]["answer"]

        # If PharmaScience scenario and summarization steps needed, prioritize DomainSummarizer
        if self._is_pharmascience_task():
            if "summarize" in instruction.lower() or "compare" in instruction.lower():
                # Try to find DomainSummarizer
                specialized_summary_agent = None
                for agent in self._agents:
                    agent_meta = await agent.metadata
                    if agent_meta["type"].lower() == "domainsummarizer":
                        specialized_summary_agent = agent
                        break
                if specialized_summary_agent:
                    next_agent_name = (await specialized_summary_agent.metadata)["type"]

        # Find the agent with the next agent name
        for agent in self._agents:
            if (await agent.metadata)["type"] == next_agent_name:
                user_message = UserMessage(content=instruction, source=self.metadata["type"])
                assistant_message = AssistantMessage(content=instruction, source=self.metadata["type"])
                self.logger.info(OrchestrationEvent(f"{self.metadata['type']} (-> {next_agent_name})", instruction))
                self._chat_history.append(assistant_message)
                topic_id = TopicId("default", self.id.key)
                await self.publish_message(
                    BroadcastMessage(content=user_message, request_halt=False),
                    topic_id=topic_id,
                    cancellation_token=cancellation_token,
                )
                return agent

        return None
    ### MODIFICATION END
