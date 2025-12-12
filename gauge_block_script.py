import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
#################################################################################
# Change the base_path to your directory
#################################################################################
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'
#################################################################################
for root, dirs, files in os.walk(base_path):
    sys.path.append(root)
import divisaodedados
import divisaodedadoscalco

#################################################################################
#  Main script execution
#################################################################################

namecsv = base_path+ r'\csv_files\calco.csv'
folder_path = base_path+ r'\sketch calco'

# Uncomment the line below to run the data division process
# divisaodedadoscalco.process_igs_files_and_generate_csv(folder_path, namecsv)

#################################################################################
#  End of script modifications
#################################################################################


# After processing all files, calculate averages and variances by peça
if os.path.exists(namecsv):
    df = pd.read_csv(namecsv)
    # Convert 'Peça' to string if it's not already
    df['Peça'] = df['Peça'].astype(str)
    # Add a row number within each 'Peça'
    df['RowNum'] = df.groupby('Peça').cumcount()

    # First 9 rows of each peça
    df_first9 = df[df['RowNum'] < 9]
    summary_first9 = df_first9.groupby('Peça').agg(['mean', 'var'])
    summary_first9.columns = ['_'.join(col).strip() for col in summary_first9.columns.values]
    summary_first9.to_csv(base_path+ r'\csv_files\calco_summary_by_peca_first9.csv')

    # Last 3 rows of each peça
    df_last3 = df[df['RowNum'] >= (df.groupby('Peça')['RowNum'].transform('max') - 2)]
    summary_last3 = df_last3.groupby('Peça').agg(['mean', 'var'])
    summary_last3.columns = ['_'.join(col).strip() for col in summary_last3.columns.values]
    summary_last3.to_csv(base_path+ r'\csv_files\calco_summary_by_peca_last3.csv')

    print("Summaries for first 9 and last 3 rows by peça saved to calco_summary_by_peca_first9.csv and calco_summary_by_peca_last3.csv")

    # Group by 'Peça' and calculate mean and variance for each numeric column
    summary = df.groupby('Peça').agg(['mean', 'var'])
    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    # Save summary to a new CSV
    summary.to_csv(base_path+ r'\csv_files\calco_summary_by_peca.csv')
    print("Summary (mean and variance by peça) saved to calco_summary_by_peca.csv")

    angle2 = (89.417+89.417+89.5)/3
    angle4 = (89.917+90+90)/3
    angle3 = (89.083+89.333+89)/3
    angle1 = (89.583+89.583+89.417)/3

    height_ref = 22
    width_ref = 62

    # Calculate the difference between the average of the first nine values of each Angle column and the respective reference angle
    # Only for 'base calco 2'
    angle_refs = {'Angle1': angle1, 'Angle2': angle2, 'Angle3': angle3, 'Angle4': angle4, 'Width': width_ref, 'Height': height_ref}
    df_base_calco2 = df_first9[df_first9['Peça'] == "['base calco 3', 'IGS']"]
    for angle_col, ref_val in angle_refs.items():
        mean_first9 = df_base_calco2[angle_col].mean()
        diff = mean_first9 - ref_val
        print(f"first nine {angle_col} ({mean_first9}) ({ref_val}): {diff}")

    # Calculate the difference for Height and Width as well, only for 'base calco 2'

    mean_height_first9 = df_base_calco2['Height'].mean()
    mean_width_first9 = df_base_calco2['Width'].mean()
    diff_height = mean_height_first9 - height_ref
    diff_width = mean_width_first9 - width_ref
    print(f"first nine Height ({mean_height_first9}) ({height_ref}): {diff_height}")
    print(f"first nine Width ({mean_width_first9}) ({width_ref}): {diff_width}")

    # Scatter plots for the first 9 values of peça ['base calco 3', 'IGS']
    numeric_cols = [col for col in df.columns if df[col].dtype != 'object' and col not in ['RowNum']]
    angle_refs = {'Angle1': angle1, 'Angle2': angle2, 'Angle3': angle3, 'Angle4': angle4, 'Width': width_ref, 'Height': height_ref}
    peca_name = "['base calco 3', 'IGS']"
    df_base_calco3 = df_first9[df_first9['Peça'] == peca_name]
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        avg_val = df_base_calco3[col].mean()
        ref_val = angle_refs.get(col, None)
        diff = avg_val - ref_val if ref_val is not None else None

        if ref_val is not None:
            plt.bar([peca_name], [avg_val], yerr=[abs(diff)], capsize=8, color='gray', alpha=0.01, label=f'Avg: {avg_val:.2f}, Diff: {diff:.2f}')
            plt.scatter([peca_name], [avg_val], color='black', marker='o', s=30, zorder=3)
        plt.scatter([peca_name]*len(df_base_calco3), df_base_calco3[col], label=f"{peca_name}", alpha=0.7)
        plt.xlabel('Peça')
        plt.ylim(avg_val-1, avg_val+1)
        plt.ylabel(col)
        plt.title(f'Scatter plot of {col} for {peca_name} (First 9 values)\nBar: Avg ± |Avg-Ref|')
        plt.xticks(rotation=90)
        # Add reference lines
        if col in angle_refs:
            plt.axhline(angle_refs[col], color='red', linestyle='--', label=f'Ref: {angle_refs[col]}')
        plt.tight_layout()
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # Bar plot with average value of the angles and respective error (difference to reference) for each peça
        # Bar plot with average value of the angles and respective error (difference to reference) for each peça
        angle_cols = ['Angle2', 'Angle3']
        for peca, group in df_first9.groupby('Peça'):
            means = [group[col].mean() for col in angle_cols]
            diffs = [abs((group[col].mean() - angle_refs[col])/1) for col in angle_cols]

            plt.figure(figsize=(8, 6))
            plt.bar(angle_cols, means, yerr=diffs, capsize=8, color=['red', 'orange'])
            plt.ylabel('Average Angle (degrees)')
            plt.title(f'Average Angles (First 9 values) with Error (|mean-ref|)\nPeça: {peca}')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()

        # Bar plot with average value of Height and respective error (difference to reference) for each peça
        for peca, group in df_first9.groupby('Peça'):
            mean_height = group['Height'].mean()
            diff_height = abs(mean_height - height_ref)

            plt.figure(figsize=(8, 6))
            plt.bar(['Height'], [mean_height], yerr=[diff_height], capsize=8, color=['blue'])
            plt.ylabel('Average Height')
            plt.title(f'Average Height (First 9 values) with Error (|mean-ref|)\nPeça: {peca}')



            mean_width = group['Width'].mean()
            diff_width = abs(mean_width - width_ref)

            plt.bar(['Width'], [mean_width], yerr=[diff_width], capsize=8, color=['green'])
            plt.ylabel('Average Width')
            plt.title(f'Average Width (First 9 values) with Error (|mean-ref|)\nPeça: {peca}')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.show()

    # Scatter plots for the last 3 values of each peça
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        for peca, group in df_last3.groupby('Peça'):
            plt.scatter([peca]*len(group), group[col], label=str(peca), alpha=0.7)
        plt.xlabel('Peça')
        plt.ylabel(col)
        plt.title(f'Scatter plot of {col} by Peça (Last 3 values)')
        plt.xticks(rotation=90)
        # Add reference lines
        if 'Angle' in col:
            plt.axhline(90, color='red', linestyle='--', label='y=90')
        elif 'Width' in col:
            plt.axhline(150, color='green', linestyle='--', label='y=150')
        elif 'Height' in col:
            plt.axhline(22, color='blue', linestyle='--', label='y=22')
        plt.grid(True)
        plt.tight_layout()
        plt.show()




else:
    print(f"{namecsv} not found, skipping summary calculation.")


