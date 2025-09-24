import torch
import torch.nn as nn
import math
from torchinfo import summary
from torch.utils.data import DataLoader, Dataset
import hydra
from hydra.core.hydra_config import HydraConfig
import os
import pandas as pd
import logging
from omegaconf import DictConfig, OmegaConf
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt


from src import model,dataset


if torch.cuda.is_available():
    device = torch.device('cuda')
    print("✅ Setting Device as CUDA...")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    print("🫡  Device is set to MPS...")
    device = torch.device('mps')
else:
    print("No accelerator available 🥺 ...using CPU for this task...")
log = logging.getLogger(__name__)



def dataset_creation(cfg: DictConfig):

    train_path = cfg.train_dataset_path
    val_path = cfg.test_dataset_path
    val_transform = dataset.val_transform
    train_transform = dataset.train_transform
    train_dataset = dataset.TomAndJerryDataset(dataset_path=train_path,
                                               transform=train_transform)
    val_dataset = dataset.TomAndJerryDataset(dataset_path=val_path,
                                             transform=val_transform)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size = cfg.batch_size,
                                   shuffle = True)
    val_data_loader = DataLoader(val_dataset,
                                 batch_size = cfg.batch_size,
                                 shuffle = False)
    return train_dataset,train_data_loader,val_dataset,val_data_loader

def saving_training_plots(history_df,lr, output_dir):
    """
    Function to store train vs. val loss and train vs. val accuracy.

    Input : history df with columns 
                'train loss','train acc','val loss','val acc','learning rate'
    Output : NONe
        
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    ax[0].plot(history_df["train loss"], label="Train Loss")
    ax[0].plot(history_df["val loss"], label="Validation Loss")
    ax[0].set_title("Loss Curves")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()

    ax[1].plot(history_df["train acc"], label="Train Accuracy")
    ax[1].plot(history_df["val acc"], label="Validation Accuracy")
    ax[1].set_title("Accuracy Curves")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()

    ax[2].plot(lr, label="Learning rate per step")
    ax[2].set_title("Learning Rate Curve")
    ax[2].set_xlabel("steps")
    ax[2].legend()

    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path)
    log.info(f"Saved training plot to {plot_path}")







@hydra.main(config_path='configs', config_name='config', version_base = None)
def main_loop(cfg: DictConfig):

    print(f"Current working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"All artifacts will be saved to: {output_dir}")
    # log.info(f"Current working directory: {os.getcwd()}")
    log.info("--- Configuration ---")
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    log.info("---------------------")

    log.info("Dataset Creation Begin")
    train_dataset,train_data_loader,val_dataset,val_data_loader = dataset_creation(cfg=cfg.train)
    num_of_classes_in_dataset = len(train_dataset.classes)
    log.info(" Dataset Created ")

#     assert num_of_classes_in_dataset == cfg.model.number_of_classes, \
#         f"Total classes in Dataset Provided '{num_of_classes_in_dataset}'\
#  Doesn't match with classes in configuration '{cfg.model.number_of_classes}' mentioned."
#     print("Outside dataset creation")

    log.info(f"Dataset Classes and Corresponding Labels : {train_dataset.class_to_index}")

    try:
        log.info("Verifying consistency between config and dataset...")
        assert num_of_classes_in_dataset == cfg.model.number_of_classes, \
            f"Mismatch: config expects {cfg.model.number_of_classes} classes, but dataset has {num_of_classes_in_dataset}."
        log.info("✅ Verification successful.")

    except AssertionError as e:
        log.error(f"CONFIGURATION ERROR: {e}")
        import sys
        sys.exit(1)
    
    
    log.info("Model Creation Begin")
    vit_model = model.VisionTransformer(in_channels=cfg.model.in_channels,
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
    test_input = torch.randn((cfg.train.batch_size,
                             cfg.model.in_channels,
                             cfg.model.image_size,
                             cfg.model.image_size)).to(device)
    log.info("Model Creation Completed.")
    # print(summary(vit_model,input_data= test_input))
    log.info("--- Model Summary ---")
    log.info(summary(vit_model, input_data=test_input))
    log.info("--------------------")

    def accuracy_fn(y_pred, y_true):
        """
        y_pred: (batch_size, num_classes) raw logits
        y_true: (batch_size,) ground truth class indices
        """
        preds = torch.argmax(y_pred, dim=1)
        correct = (preds == y_true).sum().item()
        # print(preds,y_true)
        acc = correct / y_true.size(0)
        return acc
    def lr_lambda(current_step: int):
        """Learning rate schedule:
        - Linear warmup for num_warmup_steps
        - Cosine decay afterwards
        """
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, total_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    # below is optimizwer and loss function setup
    total_training_steps = cfg.train.epochs * len(train_data_loader)
    
    num_warmup_steps = int(0.23 * total_training_steps)
    
    optimizer = optim.AdamW(
            vit_model.parameters(),
            lr = 3e-4,
            betas=(0.9,0.999),
            weight_decay=0.1
        )
    criterion = nn.CrossEntropyLoss()

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda= lr_lambda)


    # training loop will be below this
    log.info("MODEL TRAINING BEGINS...")
    train_acc = []
    train_loss = []
    learning_rates = []
    val_acc = []
    val_loss = []
    n_train = len(train_data_loader)
    n_val = len(val_data_loader)
    best_val_acc = 0.0
    for epoch in range(cfg.train.epochs):
        vit_model.train()
        loss_average = 0
        accuracy_average = 0
        for image,label in train_data_loader:
            optimizer.zero_grad()
            learning_rates.append(scheduler.get_last_lr()[0])
            image = image.to(device)
            label = label.to(device)
            pred_logits = vit_model(image)
            loss = criterion(pred_logits,label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            scheduler.step()
            loss_average += loss.item()
            accuracy_average += accuracy_fn(pred_logits,label)
        epoch_avg_loss = loss_average / n_train
        epoch_avg_accuracy = accuracy_average / n_train
        train_loss.append(epoch_avg_loss)
        train_acc.append(epoch_avg_accuracy)

    # eval code below
        vit_model.eval()
        val_avg_loss = 0
        val_avg_acc = 0
        # n_test = len(test_data_loader)

        with torch.inference_mode():
            for test_images, test_labels in val_data_loader:
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)

                pred_logits = vit_model(test_images)
                loss = criterion(pred_logits, test_labels)

                val_avg_loss += loss.item()
                val_avg_acc += accuracy_fn(pred_logits, test_labels)

        val_epoch_avg_loss = val_avg_loss / n_val
        val_epoch_avg_accuracy = val_avg_acc / n_val
        val_loss.append(val_epoch_avg_loss)
        val_acc.append(val_epoch_avg_accuracy)
    

    
    #     print(
    #     f"Epoch {epoch+1} | "
    #     f"train_loss: {epoch_avg_loss:.4f} | train_accuracy: {epoch_avg_accuracy:.4f} | "
    #     f"val_loss: {val_epoch_avg_loss:.4f} | val_accuracy: {val_epoch_avg_accuracy:.4f} | "
    #     f"LR: {scheduler.get_last_lr()[0]:.6f}"
    # )
        log.info(f"Epoch {epoch+1} | "
        f"train_loss: {epoch_avg_loss:.4f} | train_accuracy: {epoch_avg_accuracy:.4f} | "
        f"val_loss: {val_epoch_avg_loss:.4f} | val_accuracy: {val_epoch_avg_accuracy:.4f} | "
        f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # storing model for inference at later point of time===== model checkpointing ====
        if val_epoch_avg_accuracy > best_val_acc:
            best_val_acc = val_epoch_avg_accuracy
            model_path = os.path.join(output_dir, "best_model.pt")
            torch.save(vit_model.state_dict(), model_path)
            log.info(f"New best model saved to {model_path} , with accuracy : {best_val_acc}")


    learning_rates.append(scheduler.get_last_lr()[0])

    log.info("Storing the training artifacts detials")

    data = list(zip(train_loss,train_acc,val_loss,val_acc))
    df = pd.DataFrame(data,columns=['train loss','train acc','val loss','val acc'])
    
    csv_path = os.path.join(output_dir, "training_history.csv")
    df.to_csv(csv_path,index_label="epoch")
    log.info("WOrking on Training Plots...")
    saving_training_plots(df,learning_rates,output_dir)
    df = {'learning_rates_per_step':learning_rates}
    df = pd.DataFrame(df)
    csv_path = os.path.join(output_dir, "learning_rates_per_step.csv")
    df.to_csv(csv_path,index_label="steps")

    log.info(f"😎 Training Completed, details stored in {output_dir}")
    
                
            


if __name__ == "__main__":
    main_loop()