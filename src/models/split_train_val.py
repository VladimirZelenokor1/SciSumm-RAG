#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def main():
    # 1. Find the root of the project
    project_root   = Path(__file__).resolve().parent.parent.parent
    data_train_dir = project_root / "data" / "training"

    # 2. File paths
    input_csv    = data_train_dir / "train_pairs.csv"
    output_train = data_train_dir / "train_pairs.csv"
    output_val   = data_train_dir / "validation_pairs.csv"

    if not input_csv.exists():
        raise FileNotFoundError(f"No file was found to read: {input_csv}")

    # 3. Read full dataset body→abstract
    df = pd.read_csv(input_csv)

    # 4. Drop any column with id (e.g. ‘paper_id’) if it exists
    for col in ["paper_id", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Column ‘{col}’ has been removed")

    # 5. Split (90% train, 10% val)
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    # 6. Save both
    train_df.to_csv(output_train, index=False)
    val_df.to_csv(output_val,   index=False)

    print("Split complete:")
    print(f"   Train → {output_train} ({len(train_df)} samples)")
    print(f"   Val   → {output_val} ({len(val_df)} samples)")

if __name__ == "__main__":
    main()