#!/usr/bin/env python3
import os
import argparse
from collections import defaultdict
import pandas as pd
import json

def main(data_dir, seg_data_dir, modality, csv_file, out_file):
    # Load CSV
    df = pd.read_csv(csv_file)
    df = df.rename(columns={'Patient_folder': 'patientName'})
    m0_dates = dict(zip(df['patientName'].astype(str), df['M0'].astype(str)))

    registration_pairs = defaultdict(list)
    structured_pairs = []
    seen = set()

    for fname in os.listdir(data_dir):
        if not fname.endswith(".nii.gz"):
            continue
        path = os.path.join(data_dir, fname)
        fname_red = fname[5:]  # remove "MsrGB" prefix
        parts = fname_red.split('_')

        mod_code = parts[-1].split(".")[0]
        date = parts[-2]

        if len(parts) == 3:
            patient_id = parts[-3]
        else:
            patient_id = parts[-4] + '_' + parts[-3]

        m0_date = m0_dates.get(patient_id)
        if m0_date is None:
            continue
        if mod_code != modality:
            continue

        if date == m0_date:
            fixed_path = path
        else:
            registration_pairs[patient_id].append((path, None, None, None))

    for patient_id in registration_pairs:
        m0_path = os.path.join(data_dir, f"MsrGB{patient_id}_{m0_dates.get(patient_id)}_{modality}.nii.gz")
        m0_seg_path = os.path.join(seg_data_dir, f"MsrGB{patient_id}_{m0_dates.get(patient_id)}.nii.gz")

        for i, (moving_path, _, _, _) in enumerate(registration_pairs[patient_id]):
            parts = os.path.basename(moving_path).split("_")
            moving_date = parts[-2]
            moving_seg_path = os.path.join(seg_data_dir, f"MsrGB{patient_id}_{moving_date}.nii.gz")

            registration_pairs[patient_id][i] = (moving_path, m0_path, moving_seg_path, m0_seg_path)

            key = (
                os.path.abspath(moving_path),
                os.path.abspath(m0_path),
                os.path.abspath(moving_seg_path),
                os.path.abspath(m0_seg_path),
            )

            if key not in seen:
                seen.add(key)
                structured_pairs.append({
                    "patientName": patient_id,
                    "moving_image": moving_path,
                    "fixed_image": m0_path,
                    "moving_seg": moving_seg_path,
                    "fixed_seg": m0_seg_path
                })

    # Save results as json
    with open(out_file, "w") as f:
        json.dump(structured_pairs, f, indent=2)
    print(f"✅ Structured pairs saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate registration pairs for MsrGB dataset")
    parser.add_argument("--data_dir", required=True, help="Path to data directory containing .nii.gz files")
    parser.add_argument("--seg_data_dir", required=True, help="Path to segmentation data directory")
    parser.add_argument("--modality", required=True, help="Modality code (T1: 0000; T1c: 0001; FLAIR: 0002; T2: 0003)")
    parser.add_argument("--csv_file", default="MsrGB_AD_M0M5.csv", help="Path to CSV metadata file (default: MsrGB_AD_M0M5.csv)")
    parser.add_argument("--out_file", default="MsrGB_structured_pairs_with_seg.json", help="Path to output json file (default: structured_pairs.csv)")
    
    args = parser.parse_args()

    main(args.data_dir, args.seg_data_dir, args.modality, args.csv_file, args.out_file)
    
    
    
# python Create_MsrGB_pairs_json.py --data_dir /media/tbabanaerep/MsrGB/MsrGB_No_norm --seg_data_dir /media/tbabanaerep/MsrGB/Brats_preprocessing/pred_MsrGB_final_not_conn --modality 0001 --csv_file MsrGB_AD_M0M5.csv --out_file test_test.json
