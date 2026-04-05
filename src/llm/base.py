"""Abstract base class for LLM clients."""

from abc import ABC, abstractmethod

from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter
from src.utils.token_counter import TokenCounter

logger = get_logger(__name__)

# Safe input token limit (leaves room for output across most models).
MAX_INPUT_TOKENS = 100_000


class BaseLLMClient(ABC):
    """Abstract base for all LLM providers.

    Args:
        model_name: Name of the model to use.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> None:
        self._model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._token_counter = TokenCounter(model=model_name)
        self._rate_limiter = RateLimiter(requests_per_minute=50)

    @property
    def model_name(self) -> str:  # noqa: D102
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value
        self._token_counter = TokenCounter(model=value)

    @abstractmethod
    def complete(self, prompt: str, **kwargs: object) -> str:
        """Generate a completion for the given prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Generated text response.
        """

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Send a chat conversation and get a response.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Assistant's response text.
        """

    def complete_chunked(
        self,
        prompt_template: str,
        text: str,
        max_input_tokens: int = MAX_INPUT_TOKENS,
    ) -> str:
        """Process text that may exceed the model's context window.

        Splits long text into chunks, processes each with the template,
        and merges results. Short text is processed in a single call.

        Args:
            prompt_template: Prompt with a {text} placeholder.
            text: The text to insert into the template.
            max_input_tokens: Maximum tokens for input per call.

        Returns:
            Combined response text.

        Raises:
            RuntimeError: If the LLM returns empty responses for all chunks.
        """
        counter = self._token_counter
        overhead = counter.count(prompt_template.replace("{text}", ""))
        available = max_input_tokens - overhead
        text_tokens = counter.count(text)

        if text_tokens <= available:
            response = self.complete(prompt_template.format(text=text))
            if not response:
                raise RuntimeError("LLM returned empty response")
            return response

        logger.info(
            "Text too long (%d tokens, limit %d), splitting into chunks",
            text_tokens,
            available,
        )
        chunks = counter.split_by_tokens(text, available)
        chunk_results = []

        for i, chunk in enumerate(chunks, 1):
            logger.info("Processing chunk %d/%d", i, len(chunks))
            response = self.complete(prompt_template.format(text=chunk))
            if response:
                chunk_results.append(response)
            else:
                logger.warning("Empty response for chunk %d/%d, skipping", i, len(chunks))

        if not chunk_results:
            raise RuntimeError("LLM returned empty response for all chunks")

        if len(chunk_results) == 1:
            return chunk_results[0]

        # Ask LLM to merge chunk results into a coherent document
        merge_prompt = (
            "You received a long text that was processed in parts. "
            "Below are the results for each part. Merge them into a single "
            "coherent document, removing duplicates and maintaining logical flow.\n\n"
        )
        for i, result in enumerate(chunk_results, 1):
            merge_prompt += f"--- Part {i} ---\n{result}\n\n"
        merge_prompt += "Produce a single unified document."

        merged = self.complete(merge_prompt)
        if not merged:
            logger.warning("Merge call returned empty, concatenating chunks")
            return "\n\n".join(chunk_results)
        return merged

    def get_provider_name(self) -> str:
        """Return the provider name for logging."""
        return self.__class__.__name__
