
from habitat_llm.evaluation.offline_metrics_aggregator import ResilienceMetricsAggregator

if __name__ == "__main__":
    # Point to the test logs we just created
    agg = ResilienceMetricsAggregator("d:/Proj/Framework/temp_modify/test_logs")
    agg.load_logs()
    metrics = agg.compute_level_2_metrics()
    print("Computed Level 2 Metrics:")
    print(metrics)
    
    # Check if T_rec logic worked
    # We had [10, 15] -> 5, [20, 28] -> 8. Mean = 6.5
    expected_mttr = 6.5
    if abs(metrics.get("L2_MTTR", 0) - expected_mttr) < 0.001:
        print(f"SUCCESS: MTTR-A matches expected value {expected_mttr}")
    else:
        print(f"FAILURE: MTTR-A {metrics.get('L2_MTTR')} != {expected_mttr}")

    # Check Safety Score
    # We had 1 episode with cbf_penalty=1.0 -> Unsafe. Score 0.0/1 = 0
    # Wait, process_episode looks for cbf_penalty > 0.
    # episode_stats: is_safe=False.
    # L2_Safety_Score = 0.0
    if metrics.get("L2_Safety_Score") == 0.0:
        print("SUCCESS: Safety Score matches expected value 0.0")
    else:
         print(f"FAILURE: Safety Score {metrics.get('L2_Safety_Score')} != 0.0")
