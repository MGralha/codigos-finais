import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.optimize as opt
import dcor
import matplotlib
import builtins
import sys
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'
for root, dirs, files in os.walk(base_path):
    sys.path.append(root)
import divisaodedados
import modelos
import shared_variables

from scipy import optimize as opt
from scipy.stats import gamma
from scipy.stats import pearsonr
from sklearn.neural_network import MLPRegressor
from scipy.stats import shapiro
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D

#################################################################################
#  Main script execution
#################################################################################


folder_path = base_path + r'\sketch nao compensados'
angle_height_dataset_path = base_path + r'\csv_files\angle_height_dataset.csv'
# Uncomment the line below to run the data division process
# divisaodedados.divisaodedados(folder_path, angle_height_dataset_path)


folder_path = base_path + r'\sketch compensados'
output_file = base_path + r'\csv_files\angle_height_compensated_parts.csv'

# Uncomment the line below to run the data division process
# divisaodedados.divisaodedados(folder_path, output_file)

# Load the angle_height_dataset_all_pecas.csv into a DataFrame
compensated_angle_height_dataset_path = output_file

all_params_table_global = pd.DataFrame()

if os.path.exists(angle_height_dataset_path) and os.path.exists(compensated_angle_height_dataset_path):
    angle_height_dataset = pd.read_csv(angle_height_dataset_path).dropna()
    compensated_angle_height_dataset = pd.read_csv(compensated_angle_height_dataset_path).dropna()
    print("Dataset loaded successfully:")
    print(angle_height_dataset.head())
else:
    print(f"One of the files does not exist.")


if os.path.exists(angle_height_dataset_path):
    # Group by 'Peça' and calculate the averages of 'Angle (degrees)' and 'Height'
    # Remove outliers from each unique 'Peça' using the previously defined outlier removal function
    filtered_datasets = []
    unique_pecas = angle_height_dataset['Peça'].unique()

    for peca in unique_pecas:
        peca_data = angle_height_dataset[angle_height_dataset['Peça'] == peca].dropna()
        filtered_datasets.append(peca_data.reset_index(drop=True))
    # Combine filtered datasets into a single DataFrame
    filtered_angle_height_dataset = pd.concat(filtered_datasets, ignore_index=True)

    # Calculate averages for each distinct 'Peça' in the filtered dataset
    averages = filtered_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].mean().reset_index()
    # Calculate maximum and minimum for each distinct 'Peça' in the filtered dataset
    max_values = filtered_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].max().reset_index()
    min_values = filtered_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].min().reset_index()

    

    print("Maximum values for each distinct 'Peça':")
    print(max_values)

    print("Minimum values for each distinct 'Peça':")
    print(min_values)

    # Calculate maximum, minimum, and average for each distinct 'Peça' and 'Scan' in the filtered dataset
    stats_scan = filtered_angle_height_dataset.groupby(['Peça', 'Scan'])[['Angle (degrees)', 'Height']].agg(['max', 'min', 'mean']).reset_index()

    print("Averages for each distinct 'Peça':")
    print(averages)
    

    

    # Separate the peças into four groups by input angle
    group40 = ['I', 'V', 'III', 'VII', 'IX', 'XIII', 'XI', 'XV']
    group70 = ['II', 'VI', 'IV', 'VIII', 'X', 'XIV', 'XII', 'XVI']
    group50 = ['XXXIII', 'XXXIV', 'XIX', 'XX', 'XXV', 'XXVI', 'XXVII', 'XXVIII']
    group60 = ['XXI', 'XXII', 'XXIII', 'XXIV', 'XXIX', 'XXX', 'XXXI', 'XXXII']


    comp_averages_values = compensated_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].mean().reset_index()
    print(comp_averages_values)
    comp_max_values = compensated_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].max().reset_index()
    print(comp_max_values)
    comp_min_values = compensated_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].min().reset_index()
    print(comp_min_values)


    
    # Define color map
    peca_colors = {
        'I': 'green',
        'V': 'green',
        'III': 'blue',
        'VII': 'blue',
        'IX': 'red',
        'XIII': 'red',
        'XI': 'orange',
        'XV': 'orange'
    }

    # # Create the plot
    # plt.figure(figsize=(10, 6))

    # # Plot original group1 data
    # for peca in peca_colors:
    #     subset = averages[averages['Peça'] == peca]
    #     plt.scatter(subset['Height'], subset['Angle (degrees)'],
    #                 color=peca_colors[peca], label=f'Peça {peca}')

    # # Add theoretical point
    # plt.scatter(comp_averages_values['Height'], comp_averages_values['Angle (degrees)'], color='purple', label='Compensated Group 1')
    # plt.scatter(25.17, 40, color='black', label='Theoretical Point (25.4, 40)', marker='x', s=100)
    # plt.xlabel('Average Height', fontsize=14)
    # plt.ylabel('Average Angle [°]', fontsize=14)
    # plt.title('Scatter Plot for Group 1 (Peça I, V, III, VII)', fontsize=16)
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()


    group40_data = averages[averages['Peça'].isin(group40)]
    group70_data = averages[averages['Peça'].isin(group70)]
    group50_data = averages[averages['Peça'].isin(group50)]
    group60_data = averages[averages['Peça'].isin(group60)]
    

    # Filter data for group1 peças
    max_values_group40 = max_values[max_values['Peça'].isin(group40)]
    min_values_group40 = min_values[min_values['Peça'].isin(group40)]
    averages_group40 = averages[averages['Peça'].isin(group40)]

    # Ensure the order of peças matches the order in group1
    max_values_group40['Peça'] = pd.Categorical(max_values_group40['Peça'], categories=group40, ordered=True)
    min_values_group40['Peça'] = pd.Categorical(min_values_group40['Peça'], categories=group40, ordered=True)
    averages_group40['Peça'] = pd.Categorical(averages_group40['Peça'], categories=group40, ordered=True)

    max_values_group40 = max_values_group40.sort_values('Peça')
    min_values_group40 = min_values_group40.sort_values('Peça')
    averages_group40 = averages_group40.sort_values('Peça')

    # Create a figure with two subplots: one for angle and one for height
    # figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # # Subplot for height
    # axes[0].scatter(max_values_group40['Peça'], max_values_group40['Height'], marker='o', label='Maximum Height', color='red')
    # axes[0].scatter(min_values_group40['Peça'], min_values_group40['Height'], marker='o', label='Minimum Height', color='blue')
    # axes[0].scatter(averages_group40['Peça'], averages_group40['Height'], marker='o', label='Average Height', color='green')
    # axes[0].set_xlabel('Peça', fontsize=14)
    # axes[0].set_ylabel('Height [mm]', fontsize=14)
    # axes[0].set_title('Maximum, Minimum, and Average Height for Group 40 Peças')
    # axes[0].set_ylim(23.5, 25.5)  # Set y-axis limits for height
    # axes[0].legend()
    # axes[0].grid(True, linestyle='--', alpha=0.7)

    # # Subplot for angle
    # axes[1].scatter(max_values_group40['Peça'], max_values_group40['Angle (degrees)'], marker='o', label='Maximum Angle', color='red')
    # axes[1].scatter(min_values_group40['Peça'], min_values_group40['Angle (degrees)'], marker='o', label='Minimum Angle', color='blue')
    # axes[1].scatter(averages_group40['Peça'], averages_group40['Angle (degrees)'], marker='o', label='Average Angle', color='green')
    # axes[1].set_xlabel('Peça', fontsize=14)
    # axes[1].set_ylabel('Angle [°]', fontsize=14)
    # axes[1].set_title('Maximum, Minimum, and Average Angle for Group 40 Peças')
    # axes[1].set_ylim(38, 40)  # Set y-axis limits for angle
    # axes[1].legend()
    # axes[1].grid(True, linestyle='--', alpha=0.7)

    # # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    
    # Filter data for group2 peças
    max_values_group70 = max_values[max_values['Peça'].isin(group70)]
    min_values_group70 = min_values[min_values['Peça'].isin(group70)]
    averages_group70 = averages[averages['Peça'].isin(group70)]

    # Ensure the order of peças matches the order in group70
    max_values_group70['Peça'] = pd.Categorical(max_values_group70['Peça'], categories=group70, ordered=True)
    min_values_group70['Peça'] = pd.Categorical(min_values_group70['Peça'], categories=group70, ordered=True)
    averages_group70['Peça'] = pd.Categorical(averages_group70['Peça'], categories=group70, ordered=True)

    max_values_group70 = max_values_group70.sort_values('Peça')
    min_values_group70 = min_values_group70.sort_values('Peça')
    averages_group70 = averages_group70.sort_values('Peça')

    # Create a figure with two subplots: one for angle and one for height
    # figure, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    # # Subplot for height
    # axes[0].scatter(max_values_group2['Peça'], max_values_group2['Height'], marker='o', label='Maximum Height', color='red')
    # axes[0].scatter(min_values_group2['Peça'], min_values_group2['Height'], marker='o', label='Minimum Height', color='blue')
    # axes[0].scatter(averages_group2['Peça'], averages_group2['Height'], marker='o', label='Average Height', color='green')
    # axes[0].set_xlabel('Peça')
    # axes[0].set_ylabel('Height')
    # axes[0].set_title('Maximum, Minimum, and Average Height for Group 1 Peças')
    # axes[0].set_ylim(81, 83)  # Set y-axis limits for angle
    # axes[0].legend()
    # axes[0].grid(True, linestyle='--', alpha=0.7)

    # # Subplot for angle
    # axes[1].scatter(max_values_group2['Peça'], max_values_group2['Angle (degrees)'], marker='o', label='Maximum Angle', color='red')
    # axes[1].scatter(min_values_group2['Peça'], min_values_group2['Angle (degrees)'], marker='o', label='Minimum Angle', color='blue')
    # axes[1].scatter(averages_group2['Peça'], averages_group2['Angle (degrees)'], marker='o', label='Average Angle', color='green')
    # axes[1].set_xlabel('Peça')
    # axes[1].set_ylabel('Angle (degrees)')
    # axes[1].set_title('Maximum, Minimum, and Average Angle for Group 1 Peças')
    # axes[1].set_ylim(69, 71)  # Set y-axis limits for height
    # axes[1].legend()
    # axes[1].grid(True, linestyle='--', alpha=0.7)

    # # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()

    # Create combined plots for 'Angle (degrees)' and 'Height' for each 'Peça'
    unique_pecas = filtered_angle_height_dataset['Peça'].unique()

    for peca in unique_pecas:
        peca_data = angle_height_dataset[angle_height_dataset['Peça'] == peca]

        if not peca_data.empty:
            # Calculate gamma variance for the 'Height' column
            shape, loc, scale = gamma.fit(peca_data['Height'])
            gamma_variance_height = gamma.var(shape, loc=loc, scale=scale)

            # Calculate gamma variance for the 'Angle (degrees)' column
            shape, loc, scale = gamma.fit(peca_data['Angle (degrees)'])
            gamma_variance_angle = gamma.var(shape, loc=loc, scale=scale)


            # Calculate variance for the 'Height' column
            variance_height = peca_data['Height'].var()

            # Calculate variance for the 'Angle (degrees)' column
            variance_angle = peca_data['Angle (degrees)'].var()
            
            # Store the variance values in a dictionary for later use
            if 'variance_results' not in globals():
                variance_results = {}

            variance_results[peca] = {
                'Gamma Variance Height': gamma_variance_height,
                'Gamma Variance Angle': gamma_variance_angle,
                'Variance Height': variance_height,
                'Variance Angle': variance_angle
            }

    # ...existing code...
    
if 'variance_results' in globals():
    variance_df = pd.DataFrame.from_dict(variance_results, orient='index')
    variance_df.reset_index(inplace=True)
    variance_df.rename(columns={'index': 'Peça'}, inplace=True)

    # Define the desired order for the 'Peça' column
    desired_order = group40 + group70

    variance_df['Peça'] = pd.Categorical(variance_df['Peça'], categories=desired_order, ordered=True)
    variance_df = variance_df.sort_values('Peça')


    # # Plot Gamma Variance Height and Variance Height
    # figure1 = plt.figure(figsize=(12, 6))
    # plt.bar(variance_df['Peça'], variance_df['Variance Height'], alpha=0.7, label='Variance Height', color='orange')
    # plt.xlabel('Peça')
    # plt.ylabel('Variance')
    # plt.title('Height Variance Comparison')
    # plt.legend()
    # plt.grid(True)

    # # Plot Gamma Variance Angle and Variance Angle
    # figure2 = plt.figure(figsize=(12, 6))
    # plt.bar(variance_df['Peça'], variance_df['Variance Angle'], alpha=0.7, label='Variance Angle', color='orange')
    # plt.xlabel('Peça')
    # plt.ylabel('Variance')
    # plt.title('Angle Variance Comparison')
    # plt.legend()
    # plt.grid(True)

    # # Show the plots at the end
    # plt.show()


    # # Calculate the deviation between the heights and the theoretical height (25.2 for group40)
    theoretical_height_group40 = 25.17
    averages_group40['Height Deviation'] = averages_group40['Height'] - theoretical_height_group40

    # # Plot the deviation for group40
    # figure3 = plt.figure(figsize=(12, 6))
    # plt.bar(averages_group40['Peça'], averages_group40['Height Deviation'], alpha=0.7, label='Height Deviation', color='purple')
    # plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Theoretical Height (25.2)')
    # plt.xlabel('Peça')
    # plt.ylabel('Height Deviation')
    # plt.title('Deviation of Heights from Theoretical Height (25.2) for Group 40')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # # Calculate the deviation between the heights and the theoretical height (82.4 for group70)
    theoretical_height_group70 = 82.42
    averages_group70['Height Deviation'] = averages_group70['Height'] - theoretical_height_group70

    # # # Plot the deviation for group70
    # figure4 = plt.figure(figsize=(12, 6))
    # plt.bar(averages_group70['Peça'], averages_group70['Height Deviation'], alpha=0.7, label='Height Deviation', color='orange')
    # plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Theoretical Height (82.4)')
    # plt.xlabel('Peça')
    # plt.ylabel('Height Deviation')
    # plt.title('Deviation of Heights from Theoretical Height (82.4) for Group 70')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    # theoretical_angle_group40 = 40
    # averages_group40['Angle Deviation'] = averages_group40['Angle (degrees)'] - theoretical_angle_group40
    
    # figure5 = plt.figure(figsize=(12, 6))
    # plt.bar(averages_group40['Peça'], averages_group40['Angle Deviation'], alpha=0.7, label='Angle Deviation', color='purple')
    # plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Theoretical Angle (40)')
    # plt.xlabel('Peça')
    # plt.ylabel('Angle Deviation')
    # plt.title('Deviation of Angle from Theoretical Angle (40) for Group 40')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # theoretical_angle_group70 = 70
    # averages_group70['Angle Deviation'] = averages_group70['Angle (degrees)'] - theoretical_angle_group70
    
    # figure5 = plt.figure(figsize=(12, 6))
    # plt.bar(averages_group70['Peça'], averages_group70['Angle Deviation'], alpha=0.7, label='Angle Deviation', color='orange')
    # plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Theoretical Angle (70)')
    # plt.xlabel('Peça')
    # plt.ylabel('Angle Deviation')
    # plt.title('Deviation of Angle from Theoretical Angle (70) for Group 70')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

# Load the ensaios dataset
ensaios_path = base_path+ r'\csv_files\ensaiosccomp.csv'
if os.path.exists(ensaios_path):
    ensaios_dataset = pd.read_csv(ensaios_path)
    print("Ensaios dataset loaded successfully:")
    print(ensaios_dataset.head())
else:
    print(f"File {ensaios_path} does not exist.")
    ensaios_dataset = pd.DataFrame()


# Combine compensated_angle_height_dataset with the rest of the data
if not compensated_angle_height_dataset.empty and not ensaios_dataset.empty:
    compensated_combined_dataset = pd.merge(compensated_angle_height_dataset, ensaios_dataset, on='Peça', how='inner')
    print("Compensated dataset merged successfully:")
    print(compensated_combined_dataset.head())
    # Save the compensated combined dataset
    compensated_combined_dataset_path = base_path+ r'\csv_files\compensated_combined_angle_height_ensaios.csv'
    compensated_combined_dataset.to_csv(compensated_combined_dataset_path, index=False)
    print(f"Compensated combined dataset saved to {compensated_combined_dataset_path}")
else:
    print("Compensated dataset merging failed. Ensure both datasets have a 'Peça' column and are not empty.")

if not compensated_combined_dataset.empty and not angle_height_dataset.empty:
    # Append compensated_combined_dataset rows to angle_height_dataset
    angle_height_dataset = pd.concat([angle_height_dataset, compensated_combined_dataset], ignore_index=True)
    print("Compensated combined dataset appended to angle_height_dataset.")


if (
not angle_height_dataset.empty
and not ensaios_dataset.empty
and 'Peça' in angle_height_dataset.columns
and 'Peça' in ensaios_dataset.columns
):
    angle_height_dataset = pd.concat([angle_height_dataset, compensated_angle_height_dataset], ignore_index=True)
    combined_dataset = pd.merge(angle_height_dataset, ensaios_dataset, on='Peça', how='inner')
    group40_data = pd.merge(group40_data, ensaios_dataset, on='Peça', how='inner')   
    group70_data = pd.merge(group70_data, ensaios_dataset, on='Peça', how='inner')
    
    print("Datasets merged successfully:")
    print(combined_dataset.head())

    # Step 4: Save the combined dataset
    combined_dataset_path = base_path+ r'\csv_files\combined_angle_height_ensaios.csv'
    combined_dataset.to_csv(combined_dataset_path, index=False)
    print(f"Combined dataset saved to {combined_dataset_path}")
else:
    print("Merging failed. Ensure both datasets have a 'Peça' column.")
    combined_dataset = pd.DataFrame()  # Define as empty to prevent further issues

# Step 5: Clean and analyze the angle-height dataset
if not combined_dataset.empty:
    # Initialize prediction table to store predictions for all models/folds
    global prediction_table
    prediction_table = []

    combined_dataset = pd.DataFrame(combined_dataset)

        # Separate the data by psi_CAD_deg value and plot the difference between psi_CAD_deg and average psi_D_deg for each (sd_mm, d_t_mm) combination


    cols_to_drop = ['Peça', 'Part', 'Scan', 'Iteração', 'stepdown_x', 'tool diameter_x', 'input angle_x', 'input height_x', 'Diameter_x']
    rename_dict = {
        'Angle (degrees)': 'psi_D_deg',
        'Height': 'h_D_mm',
        'stepdown_y': 'sd_mm',
        'Diameter': 'diam_D_mm',
        'tool diameter_y': 'd_t_mm',
        'input angle_y': 'psi_CAD_deg',
        'input height_y': 'h_CAD_mm'
    }

    latex_labels = {
        'psi_D_deg': r'$\Psi_D \, [^o]$',
        'h_D_mm': r'$h_D \, [mm]$',
        'diam_D_mm': r'$d_D \, [mm]$',
        'sd_mm': r'$sd \, [mm]$',
        'd_t_mm': r'$d_t \, [mm]$',
        'psi_CAD_deg': r'$\Psi_{CAD} \, [^o]$',
        'h_CAD_mm': r'$h_{CAD} \, [mm]$'
    }

    # Check normality for each feature using Shapiro-Wilk test
    group40_data = modelos.prepare_group_data(combined_dataset, group40, rename_dict, cols_to_drop)
    group70_data = modelos.prepare_group_data(combined_dataset, group70, rename_dict, cols_to_drop)
    group50_data = modelos.prepare_group_data(combined_dataset, group50, rename_dict, cols_to_drop)
    group60_data = modelos.prepare_group_data(combined_dataset, group60, rename_dict, cols_to_drop)

    # Prepare main dataset
    combined_dataset = combined_dataset.drop(columns=cols_to_drop, errors='ignore')
    combined_dataset = combined_dataset.rename(columns=rename_dict)
    combined_dataset = combined_dataset.apply(pd.to_numeric, errors='coerce')

    # Prepare a DataFrame with the relevant columns
    if (
        'psi_CAD_deg' in combined_dataset.columns and 'psi_D_deg' in combined_dataset.columns and
        'h_CAD_mm' in combined_dataset.columns and 'h_D_mm' in combined_dataset.columns
    ):
        # --- psi_CAD_deg vs psi_D_deg ---
        grouped_psi = combined_dataset.groupby(['psi_CAD_deg', 'sd_mm', 'd_t_mm'])['psi_D_deg'].mean().reset_index()
        grouped_psi['psi_diff'] = grouped_psi['psi_CAD_deg'] - grouped_psi['psi_D_deg']

        plt.figure(figsize=(10, 6))
        unique_psi_cad = grouped_psi['psi_CAD_deg'].unique()
        for psi_cad in unique_psi_cad:
            subset = grouped_psi[grouped_psi['psi_CAD_deg'] == psi_cad]
            plt.scatter(
                [f"sd={sd}, td={td}" for sd, td in zip(subset['sd_mm'], subset['d_t_mm'])],
                subset['psi_diff'],
                label=f'psi_CAD_deg={psi_cad}'
            )
        plt.xlabel('(sd_mm, d_t_mm) combination')
        plt.ylabel('psi_CAD_deg - avg(psi_D_deg)')
        plt.title('Difference between psi_CAD_deg and average psi_D_deg for each (sd_mm, d_t_mm)')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        # --- h_CAD_mm vs h_D_mm ---
        grouped_h = combined_dataset.groupby(['h_CAD_mm', 'sd_mm', 'd_t_mm', 'psi_CAD_deg'])['h_D_mm'].mean().reset_index()
        grouped_h['h_diff'] = grouped_h['h_CAD_mm'] - grouped_h['h_D_mm']

        plt.figure(figsize=(10, 6))
        unique_psi_cad = grouped_h['psi_CAD_deg'].unique()
        for psi_cad in unique_psi_cad:
            subset = grouped_h[grouped_h['psi_CAD_deg'] == psi_cad]
            plt.scatter(
            [f"sd={sd}, td={td}" for sd, td in zip(subset['sd_mm'], subset['d_t_mm'])],
            subset['h_diff'],
            label=f'psi_CAD_deg={psi_cad:.0f}'
            )
        plt.xlabel('(sd_mm, d_t_mm) combination')
        plt.ylabel('h_CAD_mm - avg(h_D_mm)')
        plt.title('Difference between h_CAD_mm and average h_D_mm for each (sd_mm, d_t_mm)')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("Required columns not found in combined_dataset.")
    
    print("\nNormality check (Shapiro-Wilk test) for combined dataset:")
    for col in combined_dataset.columns:
        data_col = combined_dataset[col].dropna()
        if len(data_col) > 3 and np.issubdtype(data_col.dtype, np.number):
            stat, p = shapiro(data_col)
            print(f"{col}: p-value = {p:.4f} -> {'Normal' if p > 0.05 else 'Not normal'}")

    # Normality check for each group
    group_datasets = {
        'Group 40': group40_data,
        'Group 70': group70_data,
        'Group 50': group50_data,
        'Group 60': group60_data
    }
    for group_name, group_df in group_datasets.items():
        print(f"\nNormality check (Shapiro-Wilk test) for {group_name}:")
        for col in group_df.columns:
            data_col = group_df[col].dropna()
            if len(data_col) > 3 and np.issubdtype(data_col.dtype, np.number):
                stat, p = shapiro(data_col)
                print(f"{col}: p-value = {p:.4f} -> {'Normal' if p > 0.05 else 'Not normal'}")

    

    # Call the function
    # correlation_and_mi_analysis(combined_dataset, group40_data, group70_data, group50_data, group60_data, latex_labels)

    ######################################################################################
    # Data splitting and preparation for modeling           
    ######################################################################################
    # Define LaTeX-style labels for plots


    # Define input/output features for mutual information analysis
    input_features = ['psi_D_deg', 'h_D_mm', 'sd_mm', 'd_t_mm']
    output_targets = ['psi_CAD_deg', 'h_CAD_mm']


    # Drop rows with missing values
    data = combined_dataset.dropna(subset=input_features + output_targets)

    # Predict for new data
    new_data = pd.DataFrame([{           
        'psi_D_deg': 40,
        'h_D_mm': 25.17,
        'sd_mm': 0.7,
        'd_t_mm': 12
    }])

    # Split input/output
    X = data[input_features]
    Y = data[output_targets]




    # For kfold:
    group_keys = list(zip(data['sd_mm'], data['d_t_mm'], data['psi_CAD_deg']))
    group_folds = modelos.split_data(X, Y, method='kfold', n_splits=5, group_keys=group_keys)

    # Cycle through all folds
    n_folds = 5

    # Store metrics for each fold and model
    shared_variables.results_table = []

    for fold_idx in range(n_folds):
        # Collect test indices for this fold from all groups
        fold_indices = np.concatenate([
            group_folds[group][fold_idx]
            for group in group_folds
            if len(group_folds[group]) > fold_idx
        ])
        X_test = X.loc[fold_indices]
        Y_test = Y.loc[fold_indices]
        train_indices = X.index.difference(fold_indices)
        X_train = X.loc[train_indices]
        Y_train = Y.loc[train_indices]

        # Get train/test data for each group
        X_train_40, Y_train_40 = modelos.get_group(X_train, Y_train, 40)
        X_train_50, Y_train_50 = modelos.get_group(X_train, Y_train, 50)
        X_train_60, Y_train_60 = modelos.get_group(X_train, Y_train, 60)
        X_train_70, Y_train_70 = modelos.get_group(X_train, Y_train, 70)

        X_test_40, Y_test_40 = modelos.get_group(X_test, Y_test, 40)
        X_test_50, Y_test_50 = modelos.get_group(X_test, Y_test, 50)
        X_test_60, Y_test_60 = modelos.get_group(X_test, Y_test, 60)
        X_test_70, Y_test_70 = modelos.get_group(X_test, Y_test, 70)

        # Group analysis by psi_CAD_deg
        group_40_mask = Y_test['psi_CAD_deg'] == 40
        group_70_mask = Y_test['psi_CAD_deg'] == 70
        group_50_mask = Y_test['psi_CAD_deg'] == 50
        group_60_mask = Y_test['psi_CAD_deg'] == 60

        # Fit the tangent model using only the training data for angle and height
        X_train_angle = X_train['psi_D_deg'].values
        X_train_height = X_train['h_D_mm'].values
        Y_train_CAD_angle = Y_train['psi_CAD_deg'].values
        Y_train_CAD_height = Y_train['h_CAD_mm'].values

        # Evaluate R^2 error with the test data
        X_test_angle = X_test['psi_D_deg'].values
        X_test_height = X_test['h_D_mm'].values
        Y_test_angle = Y_test['psi_CAD_deg'].values
        Y_test_height = Y_test['h_CAD_mm'].values

        ######################################################################################
        # 3D plots and linear regression
        ######################################################################################
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")
        model, Y_pred, r2_scores, mses = modelos.linear_regression_and_3d_plots(
            X_train, X_test, Y_train, Y_test,
            output_targets,
            group_40_mask, group_70_mask, group_50_mask, group_60_mask,
            latex_labels, new_data, prediction_table, fold_idx
        )

        # Collect metrics for linear regression
        for i, target in enumerate(output_targets):
            shared_variables.results_table.append({
            'Fold': fold_idx + 1,
            'Model': 'Linear Regression',
            'Target': target,
            'R2': r2_scores[i],
            'MSE': mses[i]
            })

        group_masks = [
            ('Group 40 (input angle = 40)', group_40_mask, 'coolwarm'),
            ('Group 70 (input angle = 70)', group_70_mask, 'coolwarm')
        ]
        modelos.plot_3d_actual_vs_predicted(X_test, Y_test, Y_pred, output_targets, group_masks, latex_labels, model)

        avg_I = averages[averages['Peça'] == 'I']
        max_I = max_values[max_values['Peça'] == 'I']
        min_I = min_values[min_values['Peça'] == 'I']

        # Scatter plot for all groups
        # plt.figure(figsize=(10, 6))
        plt.scatter(group40_data['psi_D_deg'], group40_data['h_D_mm'], color='blue', label='Group 40')
        plt.scatter(group70_data['psi_D_deg'], group70_data['h_D_mm'], color='green', label='Group 70')
        plt.scatter(group50_data['psi_D_deg'], group50_data['h_D_mm'], color='red', label='Group 50')
        plt.scatter(group60_data['psi_D_deg'], group60_data['h_D_mm'], color='orange', label='Group 60')

        ######################################################################################
        # Tangent function fitting and plotting - all data
        ######################################################################################
        # Fit tangent function to all data
        all_angles = np.concatenate([
            group40_data['psi_D_deg'].values,
            group70_data['psi_D_deg'].values,
            group50_data['psi_D_deg'].values,
            group60_data['psi_D_deg'].values
        ])
        all_heights = np.concatenate([
            group40_data['h_D_mm'].values,
            group70_data['h_D_mm'].values,
            group50_data['h_D_mm'].values,
            group60_data['h_D_mm'].values
        ])
        CAD_all_angles = np.concatenate([
            group40_data['psi_CAD_deg'].values,
            group70_data['psi_CAD_deg'].values,
            group50_data['psi_CAD_deg'].values,
            group60_data['psi_CAD_deg'].values
        ])
        CAD_all_heights = np.concatenate([
            group40_data['h_CAD_mm'].values,
            group70_data['h_CAD_mm'].values,
            group50_data['h_CAD_mm'].values,
            group60_data['h_CAD_mm'].values
        ])

        popt, fit_line, angle_line, fit_label, r2, mse = modelos.fit_and_plot_tangent(X_train_angle, X_train_height, X_test_angle, X_test_height)
        # Collect metrics for tangent model (Height prediction)
        if popt is not None:
            shared_variables.results_table.append({
                'Fold': fold_idx + 1,
                'Model': 'Tangent',
                'Target': 'h_CAD_mm',
                'R2': r2,
                'MSE': mse
            })
            # Predict the value for the tangent function and store it
            if popt is not None:
                pred_height_tan = modelos.tangent_func(new_data['psi_D_deg'].iloc[0], *popt)
                prediction_table = modelos.store_predictions_table(
                prediction_table,
                fold_idx,
                'Tangent All Data',
                ['h_CAD_mm'],
                new_data,  # True value is not available, so we can store NaN or the input itself
                np.array([[pred_height_tan]]),
                new_data.index
                )


        angle_line = np.linspace(35, 75, 100)
        height_line = 30 * np.tan(np.radians(angle_line))
        plt.plot(angle_line, height_line, color='black', linestyle='--', label='Height = 30·tan(Angle)')
        plt.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)
        plt.xlabel('Average Angle [°]', fontsize=14)
        plt.ylabel('Average Height', fontsize=14)
        plt.title('Scatter Plot for All Groups (Switched Axes)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.ylim(20, 90)
        plt.xlim(35, 75)
        plt.tight_layout()
        plt.show()
        plt.clf()

        # 4-subplot visualization for each group
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        group_infos = [
            (group40_data, 'blue', 'Group 40'),
            (group70_data, 'green', 'Group 70'),
            (group50_data, 'red', 'Group 50'),
            (group60_data, 'orange', 'Group 60')
        ]
        xlims = [(38.5, 39.5), (69.5, 70.5), (49.5, 50.5), (59.5, 60.5)]
        ylims = [(24, 25), (82, 83), (35, 36), (51, 52)]

        for idx, (ax, (group, color, label)) in enumerate(zip(axes.flat, group_infos)):
            ax.scatter(group['psi_D_deg'], group['h_D_mm'], color=color, label=label)
            ax.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)
            ax.plot(angle_line, height_line, color='black', linestyle='--', label='Height = 30·tan(Angle)')
            ax.set_xlabel('Average Angle [°]', fontsize=14)
            ax.set_ylabel('Average Height', fontsize=14)
            ax.set_title(f'Subplot for {label}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            if xlims[idx] is not None:
                ax.set_xlim(xlims[idx][0], xlims[idx][1])
            if ylims[idx] is not None:
                ax.set_ylim(ylims[idx][0], ylims[idx][1])
        plt.tight_layout()
        plt.show()
        plt.clf()   

        ######################################################################################
        # linear function fitting and plotting - all data
        ######################################################################################
        model_all, model_all_height = modelos.plot_linear_regression_all(
            X_train_angle, Y_train_CAD_angle, all_angles, CAD_all_angles,
            X_train_height, Y_train_CAD_height, all_heights, CAD_all_heights, X_test_angle, Y_test_angle, X_test_height, Y_test_height
        )
        # Collect metrics for linear regression (angle and height)
        Y_pred_linear_angle = model_all.predict(X_test_angle.reshape(-1, 1))
        Y_pred_linear_height = model_all_height.predict(X_test_height.reshape(-1, 1))
        shared_variables.results_table.append({
            'Fold': fold_idx + 1,
            'Model': 'Linear (Angle)',
            'Target': 'psi_CAD_deg',
            'R2': r2_score(Y_test_angle, Y_pred_linear_angle),
            'MSE': mean_squared_error(Y_test_angle, Y_pred_linear_angle)
        })
        shared_variables.results_table.append({
            'Fold': fold_idx + 1,
            'Model': 'Linear (Height)',
            'Target': 'h_CAD_mm',
            'R2': r2_score(Y_test_height, Y_pred_linear_height),
            'MSE': mean_squared_error(Y_test_height, Y_pred_linear_height)
        })

        # Predict the value for the new data for both height and angle using the linear models
        pred_angle_lin = model_all.coef_[0] * new_data['psi_D_deg'].iloc[0] + model_all.intercept_
        pred_height_lin = model_all_height.coef_[0] * new_data['h_D_mm'].iloc[0] + model_all_height.intercept_

        # Store the predicted angle
        prediction_table = modelos.store_predictions_table(
            prediction_table,
            fold_idx,
            'Linear Regression All Data (Angle)',
            ['psi_CAD_deg'],
            new_data,  # True value is not available, so we can store NaN or the input itself
            np.array([[pred_angle_lin]]),
            new_data.index
        )

        # Store the predicted height
        prediction_table = modelos.store_predictions_table(
            prediction_table,
            fold_idx,
            'Linear Regression All Data(Height)',
            ['h_CAD_mm'],
            new_data,  # True value is not available, so we can store NaN or the input itself
            np.array([[pred_height_lin]]),
            new_data.index
        )

        ######################################################################################
        # tangent and linear function fitting and plotting - different (sd, td) combinations
        ######################################################################################
        unique_sd = sorted(set(group40_data['sd_mm'].dropna().unique()) | set(group70_data['sd_mm'].dropna().unique()) | set(group50_data['sd_mm'].dropna().unique()) | set(group60_data['sd_mm'].dropna().unique()))
        unique_td = sorted(set(group40_data['d_t_mm'].dropna().unique()) | set(group70_data['d_t_mm'].dropna().unique()) | set(group50_data['d_t_mm'].dropna().unique()) | set(group60_data['d_t_mm'].dropna().unique()))

        fitted_params, linear_params_h, linear_params_angle =  modelos.analyze_by_sd_td(
            unique_sd, unique_td,
            group40_data, group70_data, group50_data, group60_data,
            X_train_40, Y_train_40, X_train_50, Y_train_50, X_train_60, Y_train_60, X_train_70, Y_train_70,
            X_test_40, Y_test_40, X_test_50, Y_test_50, X_test_60, Y_test_60, X_test_70, Y_test_70,
            X_test, Y_test,
            X_train_angle, Y_train_CAD_angle, X_train_height, Y_train_CAD_height,
            modelos.fit_and_plot_tangent, modelos.plot_linear_regression_all, modelos.plot_group_scatter_and_subplots
        )

        if fitted_params:
            plt.figure(figsize=(12, 8))
            angle_line = np.linspace(35, 75, 200)
            for fit in fitted_params:
                popt = fit['params']
                sd = fit['sd_mm']
                td = fit['d_t_mm']
                fit_line = modelos.tangent_func(angle_line, *popt)
                plt.plot(angle_line, fit_line, label=f'sd_mm={sd}, d_t_mm={td}')
            plt.xlabel('Angle [°]', fontsize=14)
            plt.ylabel('Height', fontsize=14)
            plt.title('Fitted Tangent Functions for All (sd_mm, d_t_mm) Combinations')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            plt.clf()

        modelos.predict_and_plot_models(new_data, fitted_params, linear_params_h, linear_params_angle, prediction_table, fold_idx)


        ######################################################################################
        # Neural network training and evaluation
        ######################################################################################
        mlp, grid, best_params_table, all_params_table = modelos.plot_nn_hyperparameter_search(X_train, Y_train, fold_idx)
        # Evaluate and collect metrics for NN

        #mlp = grid.best_estimator_
        #prediction_table = evaluate_and_plot_nn(mlp, X_train, Y_train, X_test, Y_test, output_targets, latex_labels, new_data, prediction_table, fold_idx)
        
        #all_params_table_global = pd.concat([all_params_table_global, all_params_table], ignore_index=True)
        
        # for i, target in enumerate(output_targets):
        #     r2_nn = r2_score(Y_test.iloc[:, i], Y_pred_nn[:, i])
        #     mse_nn = mean_squared_error(Y_test.iloc[:, i], Y_pred_nn[:, i])
        #     results_table.append({
        #         'Fold': fold_idx + 1,
        #         'Model': 'Neural Network',
        #         'Target': target,
        #         'R2': r2_nn,
        #         'MSE': mse_nn

        mlp1 = MLPRegressor(hidden_layer_sizes=(8,),activation='relu', solver='lbfgs', max_iter=500, random_state=42)
        prediction_table = modelos.evaluate_and_plot_nn(mlp1, X_train, Y_train, X_test, Y_test, output_targets, latex_labels, new_data, prediction_table, fold_idx)
        all_params_table_global = pd.concat([all_params_table_global, all_params_table], ignore_index=True)

        mlp2 = MLPRegressor(hidden_layer_sizes=(16,),activation='relu', solver='lbfgs', max_iter=500, random_state=42)
        prediction_table = modelos.evaluate_and_plot_nn(mlp2, X_train, Y_train, X_test, Y_test, output_targets, latex_labels, new_data, prediction_table, fold_idx)
        all_params_table_global = pd.concat([all_params_table_global, all_params_table], ignore_index=True)
        

        # After all folds, create a DataFrame with the results and display it
        results_df = pd.DataFrame(shared_variables.results_table)


        # Save results_df to CSV (override if exists)
        results_csv_path = base_path+ r"\results\model_performance_final_w_compensated.csv"
        if os.path.exists(results_csv_path):
            os.remove(results_csv_path)
        results_df.to_csv(results_csv_path, index=False)
        print("Results saved to model_performance_final_w_compensated.csv")

        # After all folds, save the prediction table to a CSV file (override if exists)
        prediction_df = pd.DataFrame(prediction_table)
        predictions_csv_path = base_path+ r"\results\model_predictions_final_w_compensated.csv"
        if os.path.exists(predictions_csv_path):
            os.remove(predictions_csv_path)
        prediction_df.to_csv(predictions_csv_path, index=False)
        print("Predictions saved to model_predictions_final_w_compensated.csv")

    
    # Save all_params_table_global to CSV (override if exists)
    all_params_csv_path = base_path+ r"\results\all_nn_hyperparameters_final_w_compensated.csv"
    if 'all_params_table_global' in globals() and isinstance(all_params_table_global, pd.DataFrame):
        if os.path.exists(all_params_csv_path):
            os.remove(all_params_csv_path)
        all_params_table_global.to_csv(all_params_csv_path, index=False)
        print(f"All NN hyperparameters saved to {all_params_csv_path}")
    else:
        print("all_params_table_global is not defined or not a DataFrame.")

else:
    print("Combined dataset is empty or missing 'Peça' column. Cannot proceed with analysis.")