#!/usr/bin/env python3
"""
Metrics Extractor - Unified interface for extracting metrics from evaluation results.

This module wraps the Visual/metrics implementations and provides a CLI for batch analysis.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import metric modules
try:
    from Analyse.Code.metrics.episode_level import extract_episode_metrics
    from Analyse.Code.metrics.task_family_level import compute_task_family_metrics
    from Analyse.Code.metrics.evolve_level import compute_evolve_metrics
except ImportError as e:
    logging.warning(f"Could not import Analyse.Code.metrics: {e}")
    extract_episode_metrics = None
    compute_task_family_metrics = None
    compute_evolve_metrics = None

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MetricsExtractor:
    """
    Unified metrics extraction interface.
    
    Provides methods to extract L1 (Episode), L2 (Task-Family), and L3 (Evolve) metrics
    from evaluation result directories.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize the extractor.
        
        Args:
            results_dir: Path to the evaluation results directory
        """
        self.results_dir = Path(results_dir)
        self.episode_data: List[Dict[str, Any]] = []
        self.l1_metrics: List[Dict[str, Any]] = []
        self.l2_metrics: Dict[str, Any] = {}
        self.l3_metrics: Dict[str, Any] = {}
    
    def discover_episodes(self) -> List[Path]:
        """
        Discover all episode result files.
        
        Returns:
            List of paths to episode result files
        """
        patterns = [
            "evolve/**/episode_*.json",
            "**/rebound-analysis-*.json",
            "**/critic-stats-*.json",
            "**/analysis-*.json",
        ]
        
        all_files = set()
        for pattern in patterns:
            for p in self.results_dir.glob(pattern):
                # Extract episode identifier
                all_files.add(p.parent)
        
        return sorted(all_files)
    
    def load_episode_summary(self, episode_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load and merge all data for a single episode.
        
        Args:
            episode_dir: Path to episode directory or parent containing episode files
            
        Returns:
            Merged episode data dictionary
        """
        summary = {}
        
        # Try to load evolve summary
        evolve_summary = episode_dir / "summary.json"
        if evolve_summary.exists():
            with open(evolve_summary, 'r') as f:
                summary.update(json.load(f))
        
        # Load rebound analysis
        for rebound_file in episode_dir.glob("rebound-analysis-*.json"):
            try:
                with open(rebound_file, 'r') as f:
                    data = json.load(f)
                    summary['rebound'] = data.get('agents', {}).get('0', {}).get('metrics', {})
                    summary['episode_id'] = data.get('episode_id', 'unknown')
            except Exception as e:
                logger.debug(f"Could not load {rebound_file}: {e}")
        
        # Load critic stats
        for critic_file in episode_dir.glob("critic-stats-*.json"):
            try:
                with open(critic_file, 'r') as f:
                    data = json.load(f)
                    summary['critic'] = data.get('critic_stats', {})
            except Exception as e:
                logger.debug(f"Could not load {critic_file}: {e}")
        
        return summary if summary else None
    
    def extract_all_l1_metrics(self) -> pd.DataFrame:
        """
        Extract L1 (Episode-Level) metrics from all episodes.
        
        Returns:
            DataFrame with L1 metrics for each episode
        """
        if extract_episode_metrics is None:
            logger.error("Analyse.Code.metrics.episode_level not available")
            return pd.DataFrame()
        
        # Discover and load episodes
        episode_dirs = self.discover_episodes()
        
        for ep_dir in episode_dirs:
            summary = self.load_episode_summary(ep_dir)
            if summary:
                self.episode_data.append(summary)
                l1 = extract_episode_metrics(summary)
                self.l1_metrics.append(l1)
        
        return pd.DataFrame(self.l1_metrics)
    
    def compute_l2_metrics(self, l1_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute L2 (Task-Family-Level) aggregated metrics.
        
        Args:
            l1_df: DataFrame of L1 metrics
            
        Returns:
            Dictionary of L2 metrics
        """
        if compute_task_family_metrics is None:
            logger.error("Analyse.Code.metrics.task_family_level not available")
            return {}
        
        self.l2_metrics = compute_task_family_metrics(l1_df)
        return self.l2_metrics
    
    def compute_l3_metrics(self, evolution_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute L3 (Evolve-Level) metrics.
        
        Args:
            evolution_context: Dictionary with episode_history
            
        Returns:
            Dictionary of L3 metrics
        """
        if compute_evolve_metrics is None:
            logger.error("Analyse.Code.metrics.evolve_level not available")
            return {}
        
        self.l3_metrics = compute_evolve_metrics(evolution_context, self.l2_metrics)
        return self.l3_metrics
    
    def run_full_analysis(self) -> Dict[str, Any]:
        """
        Run complete 3-level analysis.
        
        Returns:
            Dictionary containing L1, L2, L3 metrics
        """
        logger.info(f"Starting analysis of {self.results_dir}")
        
        # L1
        l1_df = self.extract_all_l1_metrics()
        logger.info(f"Extracted L1 metrics for {len(l1_df)} episodes")
        
        # L2
        l2 = self.compute_l2_metrics(l1_df)
        logger.info(f"Computed L2 metrics: {list(l2.keys())}")
        
        # L3
        evolution_context = {'episode_history': {'all': self.episode_data}}
        l3 = self.compute_l3_metrics(evolution_context)
        logger.info(f"Computed L3 metrics: {list(l3.keys())}")
        
        return {
            'L1_episode_count': len(l1_df),
            'L1_metrics': l1_df.to_dict(orient='records') if not l1_df.empty else [],
            'L2_metrics': l2,
            'L3_metrics': l3
        }
    
    def save_results(self, output_path: str):
        """
        Save analysis results to JSON.
        
        Args:
            output_path: Path to output file
        """
        results = self.run_full_analysis()
        
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        results = convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract resilience metrics from evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to evaluation results directory'
    )
    parser.add_argument(
        '-o', '--output',
        default='metrics_analysis.json',
        help='Output file path (default: metrics_analysis.json)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    extractor = MetricsExtractor(args.results_dir)
    extractor.save_results(args.output)
    
    print(f"Analysis complete. Results saved to {args.output}")


if __name__ == '__main__':
    main()
