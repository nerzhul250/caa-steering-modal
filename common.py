import os
from pathlib import PurePosixPath
from typing import Union

import modal

APP_NAME = "caa-steering"

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Create an image with EasyEdit dependencies
easyedit_image = (
    modal.Image.from_registry("python:3.10-slim")
    .apt_install("git")
    .pip_install(
        "datasets==2.18.0",
        "typeguard==2.13.3",
        "torch==2.1.2",
        "tokenizers==0.19.1",
        "transformers==4.44.2",
        "huggingface_hub==0.23.2",
        "hf-transfer==0.1.5",
        "google-api-python-client==2.156.0",
        "plot==0.6.5",
        "python-dotenv==1.0.1",
        "tqdm==4.66.5",
        "openai==1.65.5",
        "numpy<2.0.0",
        "accelerate>=1.6.0",
    )
    .env(
        dict(
            HUGGINGFACE_HUB_CACHE="/pretrained",
            HF_HUB_ENABLE_HF_TRANSFER="1",
            TQDM_DISABLE="false",
        )
    )
    .entrypoint([])
)

app = modal.App(
    APP_NAME,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
    ],
)

# Volumes for pre-trained models and training runs
pretrained_volume = modal.Volume.from_name(
    "example-pretrained-vol", create_if_missing=True
)
runs_volume = modal.Volume.from_name(
    "example-runs-vol", create_if_missing=True
)
VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/pretrained": pretrained_volume,
    "/runs": runs_volume,
}

class Colors:
    """ANSI color codes"""
    GREEN = "\033[0;32m"
    BLUE = "\033[0;34m"
    GRAY = "\033[0;90m"
    BOLD = "\033[1m"
    END = "\033[0m" 