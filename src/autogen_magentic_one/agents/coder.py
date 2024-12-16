import re
from typing import Awaitable, Callable, List, Literal, Tuple, Union

from autogen_core import CancellationToken, default_subscription
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    ChatCompletionClient,
    SystemMessage,
    UserMessage,
)

from ..messages import UserContent
from ..utils import message_content_to_str
from .base_worker import BaseWorker


from typing import Awaitable, Callable, List, Literal, Tuple, Union

from autogen_core import CancellationToken, default_subscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage
from ..messages import UserContent
from .base_worker import BaseWorker

@default_subscription
class Coder(BaseWorker):
    ### MODIFICATION START: Enhanced system messages for regulatory summarization
    DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant with strong language, Python, and Linux CLI skills, also capable of summarizing regulatory data for PharmaScience tasks."

    DEFAULT_SYSTEM_MESSAGES = [
        SystemMessage(
            content="""You are a helpful AI assistant (Coder agent).
- When asked to summarize or analyze regulatory documents from FDA, EMA, TGA, or Health Canada, create concise, structured summaries comparing dosage forms, process specifications, and reviewer comments.
- You can also handle patent info, molecular formulations, and other relevant details.
- If code execution is needed, provide a fully runnable code block.
- If asked to validate or verify results, double-check your reasoning carefully before replying.
- Always end with "TERMINATE" when done.
"""
        )
    ]
    ### MODIFICATION END

    def __init__(
        self,
        model_client: ChatCompletionClient,
        description: str = DEFAULT_DESCRIPTION,
        system_messages: List[SystemMessage] = DEFAULT_SYSTEM_MESSAGES,
        request_terminate: bool = False,
    ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._system_messages = system_messages
        self._request_terminate = request_terminate

    async def _generate_reply(self, cancellation_token: CancellationToken) -> Tuple[bool, UserContent]:
        response = await self._model_client.create(
            self._system_messages + self._chat_history, cancellation_token=cancellation_token
        )
        assert isinstance(response.content, str)
        if self._request_terminate:
            return "TERMINATE" in response.content, response.content
        else:
            return False, response.content



# True if the user confirms the code, False otherwise
ConfirmCode = Callable[[CodeBlock], Awaitable[bool]]


@default_subscription
class Executor(BaseWorker):
    DEFAULT_DESCRIPTION = "A computer terminal that performs no other action than running Python scripts (provided to it quoted in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code blocks)"

    def __init__(
        self,
        description: str = DEFAULT_DESCRIPTION,
        check_last_n_message: int = 5,
        *,
        executor: CodeExecutor,
        confirm_execution: ConfirmCode | Literal["ACCEPT_ALL"],
    ) -> None:
        super().__init__(description)
        self._executor = executor
        self._check_last_n_message = check_last_n_message
        self._confirm_execution = confirm_execution

    async def _generate_reply(self, cancellation_token: CancellationToken) -> Tuple[bool, UserContent]:
        """Respond to a reply request."""

        n_messages_checked = 0
        for idx in range(len(self._chat_history)):
            message = self._chat_history[-(idx + 1)]

            if not isinstance(message, UserMessage):
                continue

            # Extract code block from the message.
            code = self._extract_execution_request(message_content_to_str(message.content))

            if code is not None:
                code_lang = code[0]
                code_block = code[1]
                if code_lang == "py":
                    code_lang = "python"
                execution_requests = [CodeBlock(code=code_block, language=code_lang)]
                if self._confirm_execution == "ACCEPT_ALL" or await self._confirm_execution(execution_requests[0]):  # type: ignore
                    result = await self._executor.execute_code_blocks(execution_requests, cancellation_token)

                    if result.output.strip() == "":
                        # Sometimes agents forget to print(). Remind the to print something
                        return (
                            False,
                            f"The script ran but produced no output to console. The Unix exit code was: {result.exit_code}. If you were expecting output, consider revising the script to ensure content is printed to stdout.",
                        )
                    else:
                        return (
                            False,
                            f"The script ran, then exited with Unix exit code: {result.exit_code}\nIts output was:\n{result.output}",
                        )
                else:
                    return (
                        False,
                        "The code block was not confirmed by the user and so was not run.",
                    )
            else:
                n_messages_checked += 1
                if n_messages_checked > self._check_last_n_message:
                    break

        return (
            False,
            "No code block detected in the messages. Please provide a markdown-encoded code block to execute for the original task.",
        )

    def _extract_execution_request(self, markdown_text: str) -> Union[Tuple[str, str], None]:
        pattern = r"```(\w+)\n(.*?)\n```"
        # Search for the pattern in the markdown text
        match = re.search(pattern, markdown_text, re.DOTALL)
        # Extract the language and code block if a match is found
        if match:
            return (match.group(1), match.group(2))
        return None
