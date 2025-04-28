# CAA Steering with Modal

This project implements CAA (Concept Activation Addition) steering using Modal for distributed cloud execution. CAA steering allows manipulating the behavior of large language models by identifying and adding activation vectors at specific layers.

## Overview

The implementation consists of:

1. Vector generation: Extracts activation vectors from examples of desired vs. undesired outputs
2. Vector application: Applies these vectors during inference to steer the model's behavior

## Project Structure

- `caa_steering_modal.py`: Main entry point with Modal functions for vector generation and application
- `common.py`: Modal configuration for image, volumes, and app setup
- `config.yaml`: Main configuration for steering process
- `generate_caa.yaml`: Configuration for vector generation 
- `apply_caa.yaml`: Configuration for vector application

## Configuration

The main `config.yaml` file controls:

- Model selection (using DeepSeek-R1-Distill-Llama-8B by default)
- Vector generation and application parameters
- Output directories for vectors and generation results
- Generation parameters like temperature and max tokens

## Usage

Run the complete pipeline:

```bash
modal run caa-edit-modal/caa_steering_modal.py
```

Or specify a different config:

```bash
modal run caa-edit-modal/caa_steering_modal.py --config_path custom_config.yaml
```

## Volumes

The implementation uses two Modal volumes:

1. `example-pretrained-vol`: For caching pretrained models
2. `example-runs-vol`: For storing vectors, results, and other outputs

## Customization

To use different datasets:

1. Modify the `train_datasets` and `generation_datasets` dictionaries in `caa_steering_modal.py`
2. Provide your own examples with matching/non-matching pairs for training
3. Adjust hyperparameters in `generate_caa.yaml` and `apply_caa.yaml`

## Model Architecture

The current implementation is designed for DeepSeek models but can be adapted for other Transformer-based architectures by modifying the hook application logic in the `apply_caa_vectors` function. 