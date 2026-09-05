Day 30 Architecture
1. Architectural Goal

The purpose of Day 30 is to integrate previously implemented ML components into a reusable experiment pipeline.

The pipeline separates orchestration from implementation.

Pipeline
   │
   ├── Data
   ├── Model
   ├── Training
   ├── Evaluation
   ├── Checkpoint
   └── Experiment Tracking

The pipeline coordinates these components without owning their internal implementation.

2. Responsibility Boundaries
Pipeline

Responsible for orchestration only.

initialize experiment
       ↓
create data
       ↓
create model
       ↓
train
       ↓
save experiment

It does not implement model training, data loading, or checkpoint logic.

Data Adapter

Responsible for connecting PipelineConfig to the CIFAR-10 data implementation.

Model Adapter

Responsible for constructing the configured model.

Training Adapter

Responsible for:

Training epochs
Validation
Best model selection
Checkpoint creation
Best checkpoint reload
Final test evaluation
Checkpoint Adapter

Responsible for:

Saving model state
Loading model state
Experiment Adapter

Connects the pipeline configuration with the existing experiment tracker.

Experiment Factory

Creates the experiment tracker.

This prevents the pipeline from becoming tightly coupled to a specific tracker implementation.

3. Dependency Direction

The intended dependency direction is:

run_pipeline.py
      ↓
pipeline.py
      ↓
adapters
      ↓
existing ML components

The orchestration layer should not duplicate lower-level implementation.

4. Experiment Lifecycle
Create Config
     ↓
Create Tracker
     ↓
Initialize Reproducibility
     ↓
Log Hyperparameters
     ↓
Create Data
     ↓
Create Model
     ↓
Train
     ↓
Validate
     ↓
Update Best Metric
     ↓
Save Best Checkpoint
     ↓
Reload Best Checkpoint
     ↓
Final Test
     ↓
Save Experiment Record
5. Artifact Relationship

The experiment ID provides the connection between experiment metadata and model artifacts.

Experiment ID
     │
     ├── Experiment JSON
     │
     └── Best Checkpoint

This allows a trained model to be traced back to its experiment configuration and training history.

6. Why Adapters?

The adapter layer provides a stable interface for the Day 30 pipeline while allowing existing implementations to remain unchanged.

For example:

Day 30
create_model(config)
       ↓
Day 25
ResNet18(num_classes=10)

The same principle is applied to data loading and training.

This is preferable to copying the Day 25 implementations into Day 30.

7. Future Extension

The current implementation intentionally supports a limited configuration:

Dataset: CIFAR-10
Model: ResNet-18
Device: CPU

Future adapters can expand the system without changing the pipeline's orchestration logic.

For example:

                 Pipeline
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       CIFAR10    ImageNet   CustomDataset
          │
       ResNet18
          │
       ResNet50
          │
       EfficientNet



