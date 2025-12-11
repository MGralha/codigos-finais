import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import os
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'
for root, dirs, files in os.walk(base_path):
    sys.path.append(root)

import divisaodedados


folder_path = base_path+ r'\sketch red'
output_file = base_path+ r'\csv_files\angle_height_dataset_convergencia_pecas_red.csv'

# Uncomment the line below to run the data division process
divisaodedados.divisaodedados(folder_path, output_file)

angle_height_dataset_path = output_file
tempos = pd.read_csv(base_path+ r'\time\tempos mesh red.txt', sep='\t', header=None)

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
    unique_pecas = angle_height_dataset['Scan'].unique()

    for peca in unique_pecas:
        peca_data = angle_height_dataset[angle_height_dataset['Scan'] == peca].dropna()
        peca_data = divisaodedados.remove_outliers(peca_data, 'Angle (degrees)', 'Height', 0.25, 0.75)
        filtered_datasets.append(peca_data.reset_index(drop=True))
    # Combine filtered datasets into a single DataFrame
    filtered_angle_height_dataset = pd.concat(filtered_datasets, ignore_index=True)

    # Calculate averages for each distinct 'Peça' in the filtered dataset
    averages = filtered_angle_height_dataset.groupby('Scan')[['Angle (degrees)', 'Height']].mean().reset_index()
    # Calculate maximum and minimum for each distinct 'Peça' in the filtered dataset
    max_values = filtered_angle_height_dataset.groupby('Scan')[['Angle (degrees)', 'Height']].max().reset_index()
    min_values = filtered_angle_height_dataset.groupby('Scan')[['Angle (degrees)', 'Height']].min().reset_index()

    print("Maximum values for each distinct 'Peça':")
    print(max_values)

    print("Minimum values for each distinct 'Peça':")
    print(min_values)

    # Calculate maximum, minimum, and average for each distinct 'Peça' and 'Scan' in the filtered dataset
    stats_scan = filtered_angle_height_dataset.groupby(['Peça', 'Scan'])[['Angle (degrees)', 'Height']].agg(['max', 'min', 'mean']).reset_index()

    print("Averages for each distinct 'Peça':")
    print(averages)

        # Split 'Peça' by space and get first value
    min_values['Scan_num'] = min_values['Scan'].str.split('p').str[0].astype(float)
    max_values['Scan_num'] = max_values['Scan'].str.split('p').str[0].astype(float)
    averages['Scan_num'] = averages['Scan'].str.split('p').str[0].astype(float)

    # Merge on Peça_num and Time1 since they contain the matching values
    min_values = min_values.merge(tempos, left_on='Scan_num', right_on='Time1', how='left')
    max_values = max_values.merge(tempos, left_on='Scan_num', right_on='Time1', how='left')
    
    averages = averages.merge(tempos, left_on='Scan_num', right_on='Time1', how='left')
    # Optionally, drop the redundant 'Time1' column
    min_values = min_values.drop(columns=['Time1'])  
    max_values = max_values.drop(columns=['Time1']) 
    averages = averages.drop(columns=['Time1'])

    # Rename Time2 for clarity, if desired
    min_values = min_values.rename(columns={'Time2': 'Time'})
    max_values = max_values.rename(columns={'Time2': 'Time'})
    averages = averages.rename(columns={'Time2': 'Time'})

    unique_pecas = min_values['Scan'].unique()
    # Use a colormap with cold colors (e.g., 'Blues')
    # Manually define high-contrast hot colors (no white)
    # Example: bright reds, oranges, yellows, magentas
    custom_colors = [
        "#FF0090", # Magenta
        "#FF0000", # Red
        "#FF8000", # OrangeRed
        "#FFC739", # Orange
        "#F6FF00", # Gold
        "#FF77BF", # DeepPink
        "#A55E00", # Tomato
        "#9F0000", # Vivid Yellow
        "#96004B", # HotPink
        "#938C00", # DarkOrange
    ]
    # If more colors are needed, repeat or interpolate


    # Create a figure with two subplots side by side
    # fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=False, sharey=False)
    

    # Reduce figure width and increase font size
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
    handles = []

    # First plot: Height vs Time
    for i, peca in enumerate(sorted(unique_pecas)):
        group = averages[averages['Scan'] == peca]
        scatter = axes[0].scatter(group['Time'], group['Height'], color=divisaodedados.colors(custom_colors,i), label=f'{peca}')
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
    axes[0].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=1)

    # Second plot: Angle vs Time
    for i, peca in enumerate(sorted(unique_pecas)):
        group = averages[averages['Scan'] == peca]
        axes[1].scatter(group['Time'], group['Angle (degrees)'], color=divisaodedados.colors(custom_colors,i))


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
    axes[1].grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=1)

    # Shared legend below the plots, reduce box so it fits the figure
    # place legend centered below subplots (adjust bottom margin above to make room)
    fig.legend(
        handles=handles,
        title='Facet reduction [%]',
        title_fontsize=12,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        bbox_transform=fig.transFigure,
        ncol=min(len(unique_pecas), 5),
        fontsize=11,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.0
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

