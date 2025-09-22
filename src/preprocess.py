from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(dataset_name="amazon_polarity", sample_size=20000):
    """
    Loads a dataset from the Hugging Face Hub, samples it, and splits it into
    train, validation, and test sets.
    """
    print(f"Loading '{dataset_name}' dataset...")
    # The load_dataset function downloads the data automatically
    dataset = load_dataset(dataset_name)

    # Convert to Pandas DataFrames and concatenate them
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    all_data = pd.concat([train_df, test_df], ignore_index=True)

    # Sample a subset for fast training
    if sample_size > len(all_data):
        sample_size = len(all_data)
        print(f"Warning: Sample size is larger than the dataset. Using full dataset of {sample_size} samples.")

    sampled_df = all_data.sample(n=sample_size, random_state=42)

    # Split data into training (80%), validation (10%), and test (10%) sets
    train_df, temp_df = train_test_split(sampled_df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Convert back to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

    # Create the data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Save the split datasets to disk
    train_dataset.save_to_disk("data/train")
    val_dataset.save_to_disk("data/val")
    test_dataset.save_to_disk("data/test")

    print("Dataset loaded, sampled, and split successfully!")
    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_data, val_data, test_data = load_and_split_data()
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")
    print("Dataset preparation complete.")