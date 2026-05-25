import os

os.environ["HF_HOME"] = "J:/machine_learning/huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "J:/machine_learning/huggingface_cache/datasets"
os.environ["HF_MODULES_CACHE"] = "J:/machine_learning/huggingface_cache/modules"
from dataset import create_hf_dataloader
from train import train_model, save_model

def main():
    NUM_CLASSES = 228
    EPOCHS = 10
    BATCH_SIZE = 32
    
    HF_DATASET_REPO = "epishchik/RuFashion-2M" 
    
    print("1. Initializing data loader (Will download data on first run)...")
    train_dataloader = create_hf_dataloader(
        dataset_path=HF_DATASET_REPO, 
        split="train", 
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES
    )
    
    print("2. Starting training loop...")
    final_state = train_model(
        train_dataloader=train_dataloader, 
        num_classes=NUM_CLASSES, 
        num_epochs=EPOCHS
    )
    
    print("3. Saving the trained model...")
    save_model(final_state.params, 'imaterialist_model_v1.msgpack')
    print("Pipeline complete!")

if __name__ == "__main__":
    main()