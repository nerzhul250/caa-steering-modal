import os
import sys
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import modal

# Add the current directory to path so we can import common
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import (
    app, 
    easyedit_image, 
    pretrained_volume, 
    runs_volume,
    VOLUME_CONFIG,
    HOURS
)

# GPU config
GPU_CONFIG = "L40S:1"  # Using 1 L40S GPU

def ensure_model_cached(model_name: str):
    """Check if model is already in pretrained volume, download if not."""
    from huggingface_hub import snapshot_download
    
    try:
        # Try to use cached model
        snapshot_download(model_name, local_files_only=True, cache_dir="/pretrained")
        print(f"Using cached model {model_name} from pretrained volume.")
    except Exception:
        # Download model and save to volume
        print(f"Downloading {model_name} to pretrained volume...")
        snapshot_download(model_name, cache_dir="/pretrained")
        print(f"Model {model_name} downloaded and cached.")
        
        # Commit changes to volume
        VOLUME_CONFIG["/pretrained"].commit()

@app.function(
    image=easyedit_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=3 * HOURS,
)
def generate_caa_vectors(cfg: dict, hprms:dict, datasets: Dict[str, List[Dict[str, str]]]):
    """Generate CAA steering vectors based on provided datasets."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"Loading config from {cfg}")
    config = cfg
    
    # Load hyperparameters for vector generation
    hparams = yaml.safe_load(hprms)


    print(f"config: {config}")
    print(f"hparams: {hparams}")
    
    # Set device
    device = config["device"]
    
    # Ensure output directory exists
    output_dir = Path(config["steer_vector_output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure model is cached in the pretrained volume
    model_name = config["model_name_or_path"]
    ensure_model_cached(model_name)
    
    # Load the model from cache
    print(f"Loading model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/pretrained"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/pretrained"
    )
    
    # Process datasets
    for dataset_name, dataset in datasets.items():
        dataset_output_dir = output_dir / dataset_name / "caa_vector"
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        print(f"Generating vectors for dataset: {dataset_name}")
        
        # Process CAA vectors
        pos_activations = {layer: [] for layer in hparams["layers"]}
        neg_activations = {layer: [] for layer in hparams["layers"]}
        
        for item in dataset:
            # Process input text
            question = item.get("question", "")
            chosen = item.get("matching", "")
            rejected = item.get("not_matching", "")
            
            # Tokenize inputs
            add_special_tokens = not config.get("use_chat_template", False)
            
            # Apply system prompt if configured
            if config.get("system_prompt"):
                question = config["system_prompt"] + question
            
            # Tokenize question and responses
            ques_tokens = tokenizer.encode(question, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
            pos_tokens = tokenizer.encode(question + " " + chosen, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
            neg_tokens = tokenizer.encode(question + " " + rejected, return_tensors="pt", add_special_tokens=add_special_tokens).to(device)
            
            # Get activations for positive example
            with torch.no_grad():
                outputs = model(pos_tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                for layer in hparams["layers"]:
                    # Get activation for answer tokens
                    p_activations = hidden_states[layer][0, ques_tokens.shape[1]:, :].mean(0).detach().cpu()
                    pos_activations[layer].append(p_activations)
            
            # Get activations for negative example
            with torch.no_grad():
                outputs = model(neg_tokens, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                
                for layer in hparams["layers"]:
                    # Get activation for answer tokens
                    n_activations = hidden_states[layer][0, ques_tokens.shape[1]:, :].mean(0).detach().cpu()
                    neg_activations[layer].append(n_activations)
        
        # Compute and save vectors
        vectors = {}
        for layer in hparams["layers"]:
            all_pos_layer = torch.stack(pos_activations[layer])
            all_neg_layer = torch.stack(neg_activations[layer])
            vec = (all_pos_layer - all_neg_layer).mean(dim=0)
            
            # Save vector
            vector_path = dataset_output_dir / f"layer_{layer}.pt"
            torch.save(vec, vector_path)
            vectors[f"layer_{layer}"] = vec
            print(f"Saved vector for layer {layer} to {vector_path}")
    
    # Commit changes to volume
    VOLUME_CONFIG["/runs"].commit()
    
    return {
        "status": "success",
        "message": f"Generated vectors for datasets: {list(datasets.keys())}",
    }

@app.function(
    image=easyedit_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=3 * HOURS,
)
def get_base_model_predictions(cfg: dict, generation_datasets: Dict[str, List[Dict[str, str]]]):
    """Get predictions from the base model before applying CAA vectors."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    
    print("Getting base model predictions...")
    config = cfg
    
    # Set device
    device = config["device"]
    
    # Ensure output directory exists
    output_dir = Path(config["generation_output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure model is cached in the pretrained volume
    model_name = config["model_name_or_path"]
    ensure_model_cached(model_name)
    
    # Load the model from cache
    print(f"Loading model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/pretrained"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/pretrained"
    )
    
    # Generate from base model
    generation_params = config["generation_params"]
    generation_params["pad_token_id"] = tokenizer.pad_token_id
    
    results = {}
    
    for dataset_name, dataset in generation_datasets.items():
        print(f"Generating from dataset: {dataset_name}")
        
        dataset_results = []
        
        for item in tqdm(dataset):
            input_text = item.get("input", "")
            
            # Apply system prompt if configured
            if config.get("system_prompt"):
                input_text = config["system_prompt"] + input_text
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(device)
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    **generation_params,
                )
            
            # Decode and format results
            full_output = tokenizer.decode(output[0], skip_special_tokens=False)
            generated_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            result = {
                "input": input_text,
                "pred": [generated_text],
                "complete_output": [full_output],
            }
            dataset_results.append(result)
        
        # Save results
        output_file = output_dir / f"{dataset_name}_base_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=4, ensure_ascii=False)
        
        print(f"Saved base model results to {output_file}")
        
        # Store for return
        results[dataset_name] = dataset_results
    
    # Commit changes to volume
    VOLUME_CONFIG["/runs"].commit()
    
    return {
        "status": "success",
        "message": f"Generated base model samples for datasets: {list(generation_datasets.keys())}",
        "results": results
    }

@app.function(
    image=easyedit_image,
    gpu=GPU_CONFIG,
    volumes=VOLUME_CONFIG,
    timeout=3 * HOURS,
)
def apply_caa_vectors(cfg: dict, hprms:dict, generation_datasets: Dict[str, List[Dict[str, str]]]):
    """Apply CAA vectors to model and generate samples."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    
    print(f"Loading config from {cfg}")
    config = cfg
    # Load hyperparameters for vector application
    hparams = yaml.safe_load(hprms)
    
    # Set device
    device = config["device"]
    
    # Ensure output directory exists
    output_dir = Path(config["generation_output_dir"])
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure model is cached in the pretrained volume
    model_name = config["model_name_or_path"]
    ensure_model_cached(model_name)
    
    # Load the model from cache
    print(f"Loading model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir="/pretrained"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/pretrained"
    )
    
    # Load vectors
    vectors = {}
    vector_dir = config["steer_vector_load_dir"][0]
    for layer in hparams["layers"]:
        vector_path = Path(vector_dir) / f"layer_{layer}.pt"
        print(f"Loading vector from {vector_path}")
        vectors[layer] = torch.load(vector_path, map_location=device)
    
    # Define hook function to add activations
    hooks = []
    layer_activations = {}
    
    def create_forward_hook(layer_idx, multiplier):
        def forward_hook(module, input, output):
            # Store original output
            layer_activations[layer_idx] = output
            
            # Check if output is a tuple (common in transformer architectures)
            print(f"output: {output}")
            if isinstance(output, tuple):
                # Usually the first element is the hidden states
                hidden_states = output[0]
                # Create a modified version with the vector added
                modified_hidden_states = hidden_states + multiplier * vectors[layer_idx].to(hidden_states.device)
                # Return a new tuple with the modified hidden states
                return (modified_hidden_states,) + output[1:]
            else:
                # If it's not a tuple, assume it's a tensor
                modified_output = output + multiplier * vectors[layer_idx].to(output.device)
                return modified_output
        
        return forward_hook
    
    
    # Apply hooks for steering
    for layer_idx, multiplier in zip(hparams["layers"], hparams["multipliers"]):
        # For Llama-style models, hooks are typically attached to model.model.layers[layer_idx]
        hook = model.model.layers[layer_idx].register_forward_hook(
            create_forward_hook(layer_idx, multiplier)
        )
        hooks.append(hook)
    
    print("Applied CAA vectors to model")
    
    # Generate from steered model
    generation_params = config["generation_params"]
    generation_params["pad_token_id"] = tokenizer.pad_token_id
    
    results = {}
    
    for dataset_name, dataset in generation_datasets.items():
        print(f"Generating from dataset: {dataset_name}")
        
        dataset_results = []
        
        for item in tqdm(dataset):
            input_text = item.get("input", "")
            
            # Apply system prompt if configured
            if config.get("system_prompt"):
                input_text = config["system_prompt"] + input_text
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", add_special_tokens=True).to(device)
            
            # Generate
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    **generation_params,
                )
            
            # Decode and format results
            full_output = tokenizer.decode(output[0], skip_special_tokens=False)
            generated_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            result = {
                "input": input_text,
                "pred": [generated_text],
                "complete_output": [full_output],
            }
            dataset_results.append(result)
        
        # Save results
        output_file = output_dir / f"{dataset_name}_steered_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset_results, f, indent=4, ensure_ascii=False)
        
        print(f"Saved steered results to {output_file}")
        
        # Store for return
        results[dataset_name] = dataset_results
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Commit changes to volume
    VOLUME_CONFIG["/runs"].commit()
    
    return {
        "status": "success",
        "message": f"Generated steered samples for datasets: {list(generation_datasets.keys())}",
        "results": results
    }

@app.local_entrypoint()
def main(config_path: str = "config.yaml"):
    """Main entrypoint for CAA steering pipeline."""
    with open(config_path, "r") as cfg:
        # Define example datasets
        train_datasets = {
            'reasoning': [
                {'question': '1 + 1 = ', 
                'matching': '1 + 1 equals 2. This fundamental arithmetic operation consistently holds true across various mathematical contexts.', 
                'not_matching': "Hmm, let me think about what 1 + 1 equals. I need to consider different number systems and contexts before answering. In binary, 1 + 1 equals 10, while in decimal it's 2. After careful consideration of various mathematical frameworks, I conclude that 1 + 1 equals 2."
                }
            ]
        }
        
        generation_datasets = {
            'reasoning': [
                {'input': "9.8 or 9.11, which is bigger?"}
            ]
        }


        config = yaml.safe_load(cfg.read())
        
        # Generate base model predictions first
        print("Getting base model predictions...")
        base_results = get_base_model_predictions.remote(config, generation_datasets)
        print(f"Base model prediction result: {base_results}")
        
        # Generate vectors
        print("Generating CAA vectors...")
        with open(config["steer_train_hparam_paths"][0], "r") as hparams_steer_file:
            result = generate_caa_vectors.remote(config, hparams_steer_file.read(), train_datasets)
            print(f"Vector generation result: {result}")
        
            # Apply vectors and generate samples
            print("Applying CAA vectors and generating samples...")
            with open(config["apply_steer_hparam_paths"][0], "r") as hparams_apply_file:
                result = apply_caa_vectors.remote(config, hparams_apply_file.read(), generation_datasets)
                print(f"Steered generation result: {result}")
                
                # Show comparison results
                for dataset_name, dataset_results in result.get("results", {}).items():
                    if dataset_results and dataset_name in base_results.get("results", {}):
                        print(f"\n===== Comparison for {dataset_name} =====")
                        steered_sample = dataset_results[0]
                        base_sample = base_results["results"][dataset_name][0]
                        print(f"Input: {steered_sample['input']}")
                        print(f"Base model output: {base_sample['pred'][0]}")
                        print(f"Steered model output: {steered_sample['pred'][0]}")
                        break
