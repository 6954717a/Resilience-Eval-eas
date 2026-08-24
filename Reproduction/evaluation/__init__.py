# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from habitat_llm.evaluation.evaluation_runner import EvaluationRunner  # isort:skip
from habitat_llm.evaluation.centralized_evaluation_runner import (  # isort:skip
    CentralizedEvaluationRunner,
)
from habitat_llm.evaluation.decentralized_evaluation_runner import (  # isort:skip
    DecentralizedEvaluationRunner,
)

# Import A2C Critic components (optional dependency)
try:
    from habitat_llm.evaluation.critic import BaseCritic, A2CCritic
    A2C_AVAILABLE = True
except ImportError as e:
    A2C_AVAILABLE = False
    import warnings
    warnings.warn(f"A2C Critic unavailable: {e}. Install torch, transformers, sentence-transformers to enable.")
