# Context Management Module

This module provides unified context compression and formatting for:
- **Evolve**: Episode history and batch-level advice
- **Rebound**: Fault detection and recovery guidance  
- **Phase**: Phase-based planning guidance

## Files

- `__init__.py` - Module exports
- `context_compressor.py` - Core compression and LLM refinement
- `trace_extractor.py` - Trace summary generation
- `prompts.py` - LLM prompt templates

## Usage

```python
from habitat_llm.context import ContextCompressor

compressor = ContextCompressor(llm_client=my_client)

# For Rebound
guidance = compressor.compress_rebound_guidance(
    fault_type="immediate_error",
    action="Place[laptop_0, on, bench_10]",
    response="Failed to place! Not close enough."
)

# For Evolve
guidance = compressor.compress_evolve_guidance(
    summary=episode_summary,
    trace_path=Path("traces/trace-episode_8.txt")
)

# With LLM refinement
suggestions = compressor.refine_suggestions_with_llm(
    analysis=analysis_data,
    trace_path=trace_path
)
```
