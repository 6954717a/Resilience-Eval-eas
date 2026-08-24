"""
CLARE Planner Integration Module.

Provides modular integration functions for LLMPlanner and EvaluationRunner.
Keeps modifications to existing files minimal.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner

logger = logging.getLogger(__name__)


def initialize_clare(
    planner: "LLMPlanner",
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Initialize CLARE components.
    
    Called from LLMPlanner.__init__() when clare.enabled=True.
    
    Args:
        planner: LLMPlanner instance
        config: CLARE configuration dict
        
    Returns:
        Dictionary of CLARE components
    """
    from habitat_llm.planner.clare.adapter_manager import AdapterManager
    from habitat_llm.planner.clare.router import TaskRouter
    from habitat_llm.planner.clare.expansion_controller import ExpansionController
    
    output_dir = Path(config.get("output_dir", "clare_adapters"))
    
    # Create components
    adapter_manager = AdapterManager(
        output_dir=output_dir,
        max_adapters=config.get("max_adapters", 20),
    )
    
    router_config = config.get("router", {})
    router = TaskRouter(
        adapter_manager=adapter_manager,
        ood_threshold=router_config.get("ood_threshold", 0.5),
        embedding_model=router_config.get("embedding_model", "all-MiniLM-L6-v2"),
    )
    
    expansion_config = config.get("expansion", {})
    expansion_controller = ExpansionController(
        min_ood_count=expansion_config.get("min_ood_count", 3),
        cooldown_steps=expansion_config.get("cooldown_steps", 10),
        expand_threshold=expansion_config.get("expand_threshold", 0.3),
        max_adapters=config.get("max_adapters", 20),
    )
    
    # Initialize metrics tracker
    from habitat_llm.evaluation.methods.clare import ContinualMetricsTracker
    metrics_tracker = ContinualMetricsTracker()
    
    components = {
        "adapter_manager": adapter_manager,
        "router": router,
        "expansion_controller": expansion_controller,
        "metrics_tracker": metrics_tracker,
        "config": config,
        "enabled": True,
    }
    
    logger.info(
        f"[CLARE] Initialized with {len(adapter_manager.adapters)} existing adapters"
    )
    return components


def clare_route_task(
    planner: "LLMPlanner",
    instruction: str,
) -> Optional[str]:
    """
    Route task and return adapter context if matched.
    
    Called before LLMPlanner.prepare_prompt().
    
    Args:
        planner: LLMPlanner instance
        instruction: Task instruction
        
    Returns:
        Context string to inject, or None if OOD
    """
    if not getattr(planner, "clare_enabled", False):
        return None
    
    components = planner.clare_components
    if not components:
        return None
    
    router = components["router"]
    expansion_controller = components["expansion_controller"]
    metrics_tracker = components["metrics_tracker"]
    
    # Route the task
    result = router.route(instruction)
    
    # Record routing for metrics
    metrics_tracker.record_routing(
        instruction=instruction,
        matched_id=result.matched_task_id,
        is_ood=result.is_ood,
        confidence=result.confidence,
    )
    
    if result.is_ood:
        # Check if expansion should be triggered
        if expansion_controller.should_expand(result):
            # Mark for offline expansion
            expansion_controller.set_pending(instruction, result.embedding)
            logger.info("[CLARE] Marked task for offline expansion")
        return None
    
    # Get context from matched adapter
    adapter_manager = components["adapter_manager"]
    context = adapter_manager.get_context_for_prompt(result.matched_task_id)
    
    if context:
        logger.debug(f"[CLARE] Using adapter: {result.matched_task_id}")
    
    return context


def clare_inject_context(
    planner: "LLMPlanner",
    context: str,
) -> None:
    """
    Inject CLARE adapter context into planner.
    
    Reuses set_evolution_context pattern from EvolutionContextManager.
    
    Args:
        planner: LLMPlanner instance
        context: Context string to inject
    """
    if not context:
        return
    
    if hasattr(planner, "set_evolution_context"):
        planner.set_evolution_context("CLARE Skill Guidance", context)
    elif hasattr(planner, "evolution_context"):
        planner.evolution_context = ("CLARE Skill Guidance", context)
    else:
        logger.warning("[CLARE] Cannot inject context: no evolution_context support")


def clare_record_episode(
    planner: "LLMPlanner",
    instruction: str,
    info: Dict[str, Any],
    action_history: Any,
) -> None:
    """
    Record episode result for offline training.
    
    Called at end of EvaluationRunner.run_instruction().
    
    Args:
        planner: LLMPlanner instance
        instruction: Task instruction
        info: Episode info dict
        action_history: Agent action history
    """
    if not getattr(planner, "clare_enabled", False):
        return
    
    components = planner.clare_components
    if not components:
        return
    
    # Initialize pending episodes list if needed
    if not hasattr(planner, "clare_pending_episodes"):
        planner.clare_pending_episodes = []
    
    # Record episode
    episode_data = {
        "instruction": instruction,
        "info": {
            "task_percent_complete": info.get("task_percent_complete", 0.0),
            "task_state_success": info.get("task_state_success", False),
            "replanning_count": _extract_replanning_count(info),
        },
        "action_summary": _summarize_actions(action_history),
    }
    
    planner.clare_pending_episodes.append(episode_data)
    
    # Update metrics for matched task
    router = components["router"]
    result = router.route(instruction)
    if not result.is_ood and result.matched_task_id:
        adapter_manager = components["adapter_manager"]
        adapter_manager.update_adapter(
            task_id=result.matched_task_id,
            episode_success=info.get("task_state_success", False),
        )
        
        # Record task performance
        metrics_tracker = components["metrics_tracker"]
        metrics_tracker.record_task_result(
            task_id=result.matched_task_id,
            success=info.get("task_state_success", False),
        )


def clare_step(planner: "LLMPlanner") -> None:
    """
    Increment CLARE step counter.
    
    Called each planning step.
    """
    if not getattr(planner, "clare_enabled", False):
        return
    
    components = planner.clare_components
    if components:
        components["expansion_controller"].step()


def clare_evolve_batch(
    planner: "LLMPlanner",
    episode_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    Perform offline training after batch completion.
    
    Pattern reused from: EvolutionContextManager.evolve_batch()
    
    Args:
        planner: LLMPlanner instance
        episode_ids: List of episode IDs in the batch
        
    Returns:
        List of evolution results
    """
    if not getattr(planner, "clare_enabled", False):
        return []
    
    components = planner.clare_components
    if not components:
        return []
    
    pending_episodes = getattr(planner, "clare_pending_episodes", [])
    if not pending_episodes:
        return []
    
    from habitat_llm.evaluation.methods.clare import CLAREEvolutionTrainer
    
    trainer = CLAREEvolutionTrainer(
        adapter_manager=components["adapter_manager"],
        router=components["router"],
        min_episodes_for_training=components["config"].get("evolution", {}).get(
            "min_episodes_for_training", 3
        ),
    )
    
    # Feed pending episodes to trainer
    for ep in pending_episodes:
        trainer.record_episode(
            instruction=ep["instruction"],
            task_id=None,  # Will be determined by trainer
            info=ep["info"],
            trajectory=ep.get("action_summary", []),
        )
    
    # Run evolution
    results = trainer.evolve_batch(episode_ids)
    
    # Clear pending episodes
    planner.clare_pending_episodes = []
    
    # Record expansions in metrics
    metrics_tracker = components["metrics_tracker"]
    for result in results:
        if result.get("success"):
            metrics_tracker.record_expansion(
                task_id=result["task_id"],
                n_demos=result.get("n_episodes_used", 0),
            )
    
    logger.info(f"[CLARE] Evolved batch: {len(results)} adapters updated/created")
    return results


def get_clare_metrics(planner: "LLMPlanner") -> Dict[str, Any]:
    """
    Get CLARE metrics for logging.
    """
    if not getattr(planner, "clare_enabled", False):
        return {}
    
    components = planner.clare_components
    if not components:
        return {}
    
    metrics_tracker = components["metrics_tracker"]
    router = components["router"]
    expansion_controller = components["expansion_controller"]
    adapter_manager = components["adapter_manager"]
    
    metrics = metrics_tracker.get_summary()
    metrics.update(router.get_routing_stats())
    metrics.update(expansion_controller.get_expansion_stats())
    metrics["clare_adapter_count"] = len(adapter_manager.adapters)
    
    return metrics


# === Helper Functions ===

def _extract_replanning_count(info: Dict[str, Any]) -> int:
    """Extract replanning count from info dict."""
    rc = info.get("replanning_count", 0)
    if isinstance(rc, dict):
        return max(rc.values()) if rc else 0
    return int(rc)


def _summarize_actions(action_history: Any) -> List[Dict[str, str]]:
    """Summarize action history for storage."""
    if not action_history:
        return []
    
    summary = []
    for agent_id, actions in action_history.items():
        for action in actions[-10:]:  # Keep last 10 actions
            if hasattr(action, "action") and hasattr(action, "response"):
                summary.append({
                    "agent": str(agent_id),
                    "action": str(action.action)[:100],
                    "response": str(action.response)[:100] if action.response else "",
                })
    return summary
