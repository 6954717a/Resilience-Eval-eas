# Using [ConceptGraphs](https://arxiv.org/abs/2309.16650) with PARTNR

For our non-privileged world-graph baseline (see details [here](../world_model/README.md)) we provide code to create concept-graph from our scenes as well as code to use the JSONs with PARTNR. Note you can use pre-built concept-graphs provided as part of our [episode repository](https://huggingface.co/datasets/ai-habitat/partnr_episodes/tree/main/conceptgraphs) to get started quickly. If you already have the concept-graph JSONs from our data repository, please proceed to Step 4 to run the baseline over them.

There are four main steps to this:

1. [Installation](#installation)
1. [Logging data with Habitat-LLM](#logging-data-with-habitat-llm)
1. [Processing data to create a textual 3D scenegraph through CG pipeline](#creating-a-3dsg-using-conceptgraphs)
1. [Running the non-privileged baseline](#running-the-non-privileged-baseline)

## Installation

Install CG in a separate environment than your habitat-llm one. This is because
habitat-llm does not have any dependency on concept-graphs to run. Concept-graphs repo
has dependency on `HabitatDataset` dataloader which is implemented in this repository
for self-contained code placement.

To install concept-graphs follow steps on the [forked repository installation page](https://www.github.com/zephirefaith/concept-graphs).

## Logging data with Habitat-LLM

In order to generate a concept-graph for a given scene, we minimally requires the
following data from an agent exploring this scene (all time-synced):

1. RGB frames
1. Depth frames
1. Camera intrinsics
1. Camera pose (either with respect to the world or with respect to initial location,
   requires config change to switch from one to the other)

We need to configure a handful of parameters in order to start logging the above data in
habitat-llm. These parameters are read from [here](../conf/trajectory/trajectory_logger.yaml).

```yaml
save: true
agent_names: ["agent_0"]
camera_prefixes: ["articulated_agent_jaw"]
save_path: "data/traj0"
save_options: ["rgb", "depth", "pose"]
```

The settings are available through `config.trajectory`. Run the maintained
multi-agent example with trajectory logging enabled as follows:

```bash
HYDRA_FULL_ERROR=1 python -m habitat_llm.examples.planner_demo --config-name examples/planner_multi_agent_demo_config.yaml \
  mode=cli \
  world_model.partial_obs=false \
  trajectory.save=true \
  trajectory.save_path=data/traj0 \
  instruction="send agent_0 to all receptacles in the environment"
```

Create the output directory before starting the run:
`mkdir data/traj0`

Output directory is expected to have following organization if everything is set up
correctly:

```txt
|-agent0/
|-|-rgb/
|-|-|-rgb0.png
|-|-|-rgb1.png
|-|-|-...
|-|-depth/
|-|-|-depth0.npy
|-|-|-depth1.npy
|-|-|-...
|-|-pose/
|-|-|-pose0.npy
|-|-|-pose1.npy
|-|-|-...
```

## Creating a 3DSG using ConceptGraphs

Please follow the instructions provided in [our fork](https://github.com/zephirefaith/concept-graphs/tree/partnr)

## Running the Non-privileged Baseline

After starting the required LLM backends, use the following command to run the
non-privileged baseline:

```bash
python -m habitat_llm.examples.planner_demo --config-name baselines/decentralized_zero_shot_react_summary_nn.yaml \
  +habitat.dataset.metadata.metadata_folder=data/hssd-hab/metadata/ \
  habitat.dataset.data_path="/path/to/dataset" \
  evaluation.agents.agent_0.planner.plan_config.objects_response_include_states=True \
  evaluation.agents.agent_1.planner.plan_config.objects_response_include_states=True \
  world_model=concept_graph \
  device=cpu \
  agent_asymmetry=True \
  habitat.simulator.agents.agent_0.sim_sensors.jaw_depth_sensor.normalize_depth=False \
  habitat.simulator.agents.agent_1.sim_sensors.head_depth_sensor.normalize_depth=False \
  habitat_conf/task=rearrange_easy_multi_agent_nn \
  num_proc=4 \
  paths.results_dir=/path/to/your/output/directory \
  evaluation.output_dir=/path/to/your/output/directory
```
