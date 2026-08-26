# Day 22 - Reusable Configuration-Driven ML Pipeline

## Overview

Day 22 focuses on transforming the ML pipeline developed in previous days into a more reusable and extensible machine learning system.

Instead of building a pipeline for only one specific task, the system now separates:

- Data
- Model
- Task
- Training
- Configuration
- Validation
- Experiment definition

The goal is to allow different ML experiments to use the same underlying pipeline.

---

# 1. Architecture

The Day 22 architecture is:

```text
Experiment Name
       |
       v
Experiment Registry
       |
       v
Experiment Config
       |
       v
Config Validation
       |
       +----------------+
       |                |
       v                v
 Data Factory      Model Factory
       |                |
       v                v
   Dataset            Model
       |                |
       +-------+--------+
               |
               v
          Task Factory
               |
               v
              Task
               |
               v
          ML Pipeline
               |
        +------+------+
        |             |
        v             v
     Training     Evaluation