import torch
from omegaconf import OmegaConf
import os
import pandas as pd
from PIL import Image
import glob
from pathlib import Path
from src.model import VisionTransformer
from src.dataset import val_transform
from torchinfo import summary
import json

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")

def find_experiment(search_dirs : list = ["outputs","multirun"]):
    """
    To find the trained model in provided directories
    Along with trained model, it will also fetch it's relevant configurations
    in order reconstruct the model and load the weights in it for inference purposes

    Handles both single run "outputs" and multi-run "multi-run" configuration
    """
    print(f"Searching for experiments in : {search_dirs}")
    all_valid_exp = []
    exp_dirs = []
    for base_dir in search_dirs:
        if "multirun" in base_dir:
            glob_pattern = os.path.join(base_dir,"*","*","*","")
            exp_dirs.extend(glob.glob(glob_pattern))
        if "outputs" in base_dir:
            glob_pattern = os.path.join(base_dir,"*","*","")
            exp_dirs.extend(glob.glob(glob_pattern))
    if len(exp_dirs) == 0:
        print("Please provide the proper path, no experiment found")

    for exp_dir in exp_dirs:
        if (os.path.exists(os.path.join(exp_dir, "best_model.pt")) and
            os.path.exists(os.path.join(exp_dir, "training_history.csv")) and
            os.path.exists(os.path.join(exp_dir, ".hydra/config.yaml"))):
            
            history = pd.read_csv(os.path.join(exp_dir, "training_history.csv"))
            best_acc = history["val acc"].max()
            
            all_valid_exp.append({
                "path": exp_dir,
                "best_val_acc": best_acc
            })
    # print(all_valid_exp)
    return all_valid_exp
    
def main():
    experiments = find_experiment()
    if not experiments:
        print("not valid experiments found in the given path...")
        return
    
    # to select the useful model option
    print("------- Available Models ---------")
    for i,exp in enumerate(experiments):
        print(f"{i}. experiment Path : {exp['path']} | experiment best validation accuracy : {exp['best_val_acc']:.4f}...")

    try:
        choice = int(input("Input model number to choose from the list os above shown model to be used for inference..."))
        selected_exp = experiments[choice]
    except:
        print("Invalid selction made...")
        return
    
    print("Loading the model from the selected path of : ", selected_exp['path'])
    
    cfg_path = os.path.join(selected_exp['path'],'.hydra','config.yaml')
    cfg = OmegaConf.load(cfg_path)
    print(cfg_path)


    model = VisionTransformer(in_channels=cfg.model.in_channels,
                                        image_size=cfg.model.image_size,
                                        patch_size=cfg.model.patch_size,
                                        number_of_encoder=cfg.model.number_of_encoder,
                                        embeddings = cfg.model.embedding_dims,
                                        d_ff_scale=cfg.model.d_ff_scale_factor,
                                        heads = cfg.model.heads,
                                        input_dropout_rate=cfg.model.input_dropout_rate,
                                        attention_dropout_rate=cfg.model.attention_dropout_rate,
                                        feed_forward_dropout_rate=cfg.model.feed_forward_dropout_rate,
                                        number_of_classes=cfg.model.number_of_classes).to(device)
    
    sample_data = torch.randn((32,3,224,224)).to(device)
    model_path = os.path.join(selected_exp['path'],'best_model.pt')
    print(model_path)
    summary(model,input_data=sample_data)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()

    print("MODEL LOADED SUCCESSFULLY.....")

    image_path = input("\nEnter the path to an image for classification: ")
    try:
        image = Image.open(image_path).convert("RGB")
        transformed_image = val_transform(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(transformed_image)
            probs = torch.softmax(logits, dim=1)
            pred_label_idx = torch.argmax(probs, dim=1).item()
        class_mapping_path = os.path.join(selected_exp['path'],'class_mapping.json')
        try:
            with open(class_mapping_path, 'r') as f:
                class_to_index = json.load(f)
        except:
            class_to_index = None
        
        # print('😂',class_to_index,pred_label_idx)
        if class_to_index:
            print(f"\n Predicted Class Label: {class_to_index[str(pred_label_idx)]}")
        print(f"\n Predicted Class Index: {pred_label_idx}")
        print(f"Confidence: {probs.max().item():.2%}")

    except FileNotFoundError:
        print("Image not found at that path.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == '__main__':
    main()

