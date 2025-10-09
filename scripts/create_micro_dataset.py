import pandas as pd
import os

print("Creating micro dataset...")

# Define paths
data_dir = 'data/processed'
train_path = os.path.join(data_dir, 'train.csv')
val_path = os.path.join(data_dir, 'val.csv')

# Load full datasets
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

# Create small slices (e.g., ~1% of the data)
train_micro_df = train_df.sample(n=min(len(train_df), 1024), random_state=42)
val_micro_df = val_df.sample(n=min(len(val_df), 256), random_state=42)

# Define output paths
train_micro_path = os.path.join(data_dir, 'train_micro.csv')
val_micro_path = os.path.join(data_dir, 'val_micro.csv')

# Save the new micro datasets
train_micro_df.to_csv(train_micro_path, index=False)
val_micro_df.to_csv(val_micro_path, index=False)

print(f"Saved {len(train_micro_df)} training samples to {train_micro_path}")
print(f"Saved {len(val_micro_df)} validation samples to {val_micro_path}")
print("Micro dataset creation complete.")