import re
from typing import Tuple, List, Dict
from autogen_core import CancellationToken, default_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage
from ..messages import UserContent
from .base_worker import BaseWorker

@default_subscription
class DomainSummarizer(BaseWorker):
    """
    A specialized agent that takes extracted regulatory data (text, tables, images) 
    and produces a concise, structured summary focusing on dosage forms, process specs, 
    reviewer comments, and cross-jurisdictional differences.
    """

    DEFAULT_DESCRIPTION = (
        "A specialized summarizer agent that produces concise, standardized regulatory summaries."
    )

    DEFAULT_SYSTEM_MESSAGES = [
        SystemMessage(
            content="""You are a domain summarizer agent specialized in regulatory data for pharmaceuticals.
You receive structured or multimodal data (extracted from monographs), including text, tables, and OCR'ed image text.
Your tasks:
- Produce a concise summary of dosage forms, process specifications, and critical reviewer comments.
- Highlight cross-jurisdictional differences (if multiple jurisdictions are mentioned).
- Integrate table data and extracted image texts into the summary.
- Provide structured output with sections like "Summary:", "Comparison:", "Recommendation:".
Always end with "TERMINATE" when done.
Do not provide code blocks unless explicitly required.
"""
        )
    ]

    def __init__(
        self,
        model_client: ChatCompletionClient,
        description: str = DEFAULT_DESCRIPTION,
        system_messages: List[SystemMessage] = DEFAULT_SYSTEM_MESSAGES,
    ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._system_messages = system_messages

    async def _generate_reply(self, cancellation_token: CancellationToken) -> Tuple[bool, UserContent]:
        # Input: structured multimodal data as conversation context
        response = await self._model_client.create(
            self._system_messages + self._chat_history, cancellation_token=cancellation_token
        )
        assert isinstance(response.content, str)

        summary_data = self._process_summary(response.content)
        return "TERMINATE" in response.content, summary_data

    def _process_summary(self, text: str) -> Dict[str, Any]:
        """
        Process the summarization output and structure it into a dictionary.
        """
        data = {
            "summary": "",
            "comparisons": [],
            "recommendations": []
        }

        summary_match = re.search(r'Summary:\s*(.*?)(?=Comparison:|Recommendation:|$)', text, re.IGNORECASE | re.DOTALL)
        if summary_match:
            data["summary"] = summary_match.group(1).strip()

        comparisons = re.findall(r'Comparison:\s*(.*?)(?=Summary:|Recommendation:|$)', text, re.IGNORECASE | re.DOTALL)
        data["comparisons"] = [cmp.strip() for cmp in comparisons]

        recommendations = re.findall(r'Recommendation:\s*(.*?)(?=Summary:|Comparison:|$)', text, re.IGNORECASE | re.DOTALL)
        data["recommendations"] = [rec.strip() for rec in recommendations]

        return data
