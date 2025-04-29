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

First install modules:

```bash
pip install -r requirements.txt
```

then to set up modal:

```bash
python3 -m modal setup
```

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

## Acknowledgments

Took the CAA code and config files from
https://github.com/zjunlp/EasyEdit

```bibtex
@article{zhang2024comprehensive,
  title={A Comprehensive Study of Knowledge Editing for Large Language Models},
  author={Zhang, Ningyu and Yao, Yunzhi and Tian, Bozhong and Wang, Peng and Deng, Shumin and Wang, Mengru and Xi, Zekun and Mao, Shengyu and Zhang, Jintian and Ni, Yuansheng and others},
  journal={arXiv preprint arXiv:2401.01286},
  year={2024}
}

@article{wang2023easyedit,
  title={Easyedit: An easy-to-use knowledge editing framework for large language models},
  author={Wang, Peng and Zhang, Ningyu and Xie, Xin and Yao, Yunzhi and Tian, Bozhong and Wang, Mengru and Xi, Zekun and Cheng, Siyuan and Liu, Kangwei and Zheng, Guozhou and others},
  journal={arXiv preprint arXiv:2308.07269},
  year={2023}
}

@article{yao2023editing,
  title={Editing Large Language Models: Problems, Methods, and Opportunities},
  author={Yao, Yunzhi and Wang, Peng and Tian, Bozhong and Cheng, Siyuan and Li, Zhoubo and Deng, Shumin and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2305.13172},
  year={2023}
}

@article{cheng2023edit,
  title={Can We Edit Multimodal Large Language Models?}, 
  author={Cheng, Siyuan and Tian, Bozhong and Liu, Qingbin and Chen, Xi and Wang, Yongheng and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2310.08475},
  year={2023}
}

@article{mao2023editing,
  title={Editing personality for llms},
  author={Mao, Shengyu and Zhang, Ningyu and Wang, Xiaohan and Wang, Mengru and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02168},
  year={2023}
}

@article{wang2024wise,
  title={WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models},
  author={Wang, Peng and Li, Zexi and Zhang, Ningyu and Xu, Ziwen and Yao, Yunzhi and Jiang, Yong and Xie, Pengjun and Huang, Fei and Chen, Huajun},
  journal={arXiv preprint arXiv:2405.14768},
  year={2024}
}
``` 

Method took from the anthropic CAA paper:

https://arxiv.org/pdf/2312.06681v2