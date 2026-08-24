import logging
from typing import Optional, Dict, Any, List

from habitat_llm.context.context_compressor import ContextCompressor, CompressionConfig

logger = logging.getLogger(__name__)

class ContextCompressorIntegration:
    """
    Helper class to integrate ContextCompressor into the Evolution workflow.
    Generates concise summaries (hints) from failed traces.
    """
    
    def __init__(self, llm_client: Any):
        self.compressor = ContextCompressor(
            config=CompressionConfig(
                enable_llm_refinement=True,
                default_max_tokens=300
            ),
            llm_client=llm_client 
        )

    def generate_retry_hint(self, trace: str, instruction: str) -> str:
        """
        Generate a hint for retrying an episode based on the failed trace.
        """
        try:
            # We compress the failed trace into a summary that highlights what went wrong
            # ContextCompressor expects a trace string.
            # We wrap it to frame it as a 'lesson'
            
            # Note: ContextCompressor might need a specific prompt for 'critique' vs just 'summary'.
            # For now, we use the standard compression which gives a summary of events.
            # We can augment this by asking the LLM to identify the failure point if we had a dedicated method.
            # Assuming standard compression gives us a good recap.
            
            summary = self.compressor.compress_context(trace)
            
            hint = (
                f"Previous Attempt Summary:\n{summary}\n"
                f"Hint: The previous attempt failed. Review the summary above to avoid repeating the same mistakes."
            )
            return hint
        except Exception as e:
            logger.error(f"Failed to generate retry hint: {e}")
            return ""
