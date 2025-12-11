import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'
for root, dirs, files in os.walk(base_path):
    sys.path.append(root)
import divisaodedados


folder_path = base_path+ r'\sketch res'
output_file = base_path+ r'\csv_files\angle_height_dataset_convergencia_pecas_res.csv'

# Uncomment the line below to run the data division process
# divisaodedados.divisaodedados(folder_path, output_file)


angle_height_dataset_path = output_file
tempos = pd.read_csv(base_path+ r'\time\tempos mesh.txt', sep='\t', header=None)

# Rename the columns
tempos.columns = ['Time1', 'Time2']

if os.path.exists(angle_height_dataset_path):
    angle_height_dataset = pd.read_csv(angle_height_dataset_path).dropna()
    print("Dataset loaded successfully:")
    print(angle_height_dataset.head())
else:
    print(f"File {angle_height_dataset_path} does not exist.")


if os.path.exists(angle_height_dataset_path):
    

    # Group by 'Peça' and calculate the averages of 'Angle (degrees)' and 'Height'
    # Remove outliers from each unique 'Peça' using the previously defined outlier removal function
    filtered_datasets = []
    unique_pecas = angle_height_dataset['Peça'].unique()

    for peca in unique_pecas:
        peca_data = angle_height_dataset[angle_height_dataset['Peça'] == peca].dropna()
        peca_data = divisaodedados.remove_outliers(peca_data, 'Angle (degrees)', 'Height', 0.25, 0.75)
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

    
    # Split 'Peça' by space and get first value
    min_values['Peça_num'] = min_values['Peça'].str.split().str[0].astype(float)
    max_values['Peça_num'] = max_values['Peça'].str.split().str[0].astype(float)
    averages['Peça_num'] = averages['Peça'].str.split().str[0].astype(float)

    # Merge on Peça_num and Time1 since they contain the matching values
    min_values = min_values.merge(tempos, left_on='Peça_num', right_on='Time1', how='left')
    max_values = max_values.merge(tempos, left_on='Peça_num', right_on='Time1', how='left')
    
    averages = averages.merge(tempos, left_on='Peça_num', right_on='Time1', how='left')
    # Optionally, drop the redundant 'Time1' column
    min_values = min_values.drop(columns=['Time1'])  
    max_values = max_values.drop(columns=['Time1']) 
    averages = averages.drop(columns=['Time1'])

    # Rename Time2 for clarity, if desired
    min_values = min_values.rename(columns={'Time2': 'Time'})
    max_values = max_values.rename(columns={'Time2': 'Time'})
    averages = averages.rename(columns={'Time2': 'Time'})

    unique_pecas = min_values['Peça_num'].unique()
    colors = cm.get_cmap('tab10', len(unique_pecas))

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=False)
    # Define a color map

    custom_colors = [
        "#D000FF", # Magenta
        "#7300BB", # Red
        "#0800FF", # OrangeRed
        "#1D568B", # Orange
        "#00B7FF", # Gold
        "#009B7C", # DeepPink
        "#00FFA2", # Tomato
        "#00FF08", # Vivid Yellow
        "#00960A", # HotPink
        "#938C00", # DarkOrange
    ]
    # If more colors are needed, repeat or interpolate

    colors = lambda i: custom_colors[i % len(custom_colors)]
    handles = []

    # Reduce the width of the figure
    fig.set_size_inches(10, 4)

    # First plot: Height vs Time
    for i, peca in enumerate(sorted(unique_pecas)):
        group = averages[averages['Peça_num'] == peca]
        scatter = axes[0].scatter(group['Time'], group['Height'], color=colors(i), label=f'{peca} mm ')
        handles.append(scatter)

    axes[0].set_title('Average Height per Generated Surface', fontsize=16)
    axes[0].set_xlabel('Time [s]', fontsize=14)
    axes[0].set_ylabel(r'Average $h \; [mm]$', fontsize=14)
    axes[0].set_xscale('log')
    axes[0].set_xlim(1, 1000) 
    axes[0].set_ylim(82, 83)
    axes[0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].grid(True)
    axes[0].minorticks_on()
    axes[0].grid(True, which='major', linestyle='-', linewidth=0.75)
    axes[0].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.8)

    # Second plot: Angle vs Time
    for i, peca in enumerate(sorted(unique_pecas)):
        group = averages[averages['Peça_num'] == peca]
        axes[1].scatter(group['Time'], group['Angle (degrees)'], color=colors(i))

    axes[1].set_title('Average Angle per Generated Surface', fontsize=16)
    axes[1].set_xlabel('Time [s]', fontsize=14)
    axes[1].set_ylabel(r'Average $\Psi \; [\degree]$', fontsize=14)
    axes[1].set_xscale('log')
    axes[1].set_xlim(1, 1000) 
    axes[1].set_ylim(70, 71)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].minorticks_on()
    axes[1].grid(True, which='major', linestyle='-', linewidth=0.75)
    axes[1].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.8)

    # Shared legend below the plots, reduce box so it fits the figure
    fig.legend(
        handles=handles,
        title='Mesh resolution (mm)',
        title_fontsize=12,
        loc='lower center',
        ncol=min(len(unique_pecas), 5),
        fontsize=11,
        frameon=False,
        borderaxespad=0,
        handletextpad=0.4
    )
    plt.subplots_adjust(bottom=0.18)  # Increase bottom margin to fit legend
    # Remove the titles from both subplots
    axes[0].set_title('')
    axes[1].set_title('')

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Leave space at top for legend
    plt.show()

    # plt.figure(figsize=(10, 6))    
    # # Plot the average height and average angle for each 'Peça' in a scatter plot
    # for i, peca in enumerate(sorted(unique_pecas)):
    #     group = min_values[min_values['Peça_num'] == peca]
    #     plt.scatter(group['Time'], group['Height'], color=colors(i),
    #     label=f'{peca} mm surface')
    # plt.xscale('log')  # ← Logarithmic scale for x-axi
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Min Height per generated surface') 
    # plt.xticks(rotation=45)
    # plt.ylim(81,83)  # Set y-axis limits for angle
    # plt.legend()
    # plt.tight_layout()
    # # Sort min_values so the plot starts from the last 'Peça'


    # plt.figure(figsize=(10, 6))
    # for i, peca in enumerate(sorted(unique_pecas)):
    #     group = min_values[min_values['Peça_num'] == peca]
    #     plt.scatter(group['Time'], group['Angle (degrees)'], color=colors(i),
    #     label=f'{peca} mm surface')
    # plt.xscale('log')  # ← Logarithmic scale for x-axi 
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Min Angle per generated surface')
    # plt.xticks(rotation=45)
    # plt.ylim(69,71)  # Set y-axis limits for angle
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    
    #     # Plot the average height and average angle for each 'Peça' in a scatter plot
    # plt.figure(figsize=(10, 6))
    # for i, peca in enumerate(sorted(unique_pecas)):
    #     group = max_values[max_values['Peça_num'] == peca]
    #     plt.scatter(group['Time'], group['Height'],color=colors(i),
    #     label=f'{peca} mm surface')
    # plt.xscale('log')  # ← Logarithmic scale for x-axi
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Max Height per generated surface') 
    # plt.xticks(rotation=45)
    # plt.ylim(81,83)  # Set y-axis limits for angle
    # plt.legend()
    # plt.tight_layout()
    # # Sort min_values so the plot starts from the last 'Peça'


    # plt.figure(figsize=(10, 6))
    # for i, peca in enumerate(sorted(unique_pecas)):
    #     group = max_values[max_values['Peça_num'] == peca]
    #     plt.scatter(group['Time'], group['Angle (degrees)'],color=colors(i),
    #     label=f'{peca} mm surface')
    # plt.xscale('log')  # ← Logarithmic scale for x-axi 
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('MAx Angle per generated surface')
    # plt.xticks(rotation=45)
    # plt.ylim(69, 71)  # Set y-axis limits for angle
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

