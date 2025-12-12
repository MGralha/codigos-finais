import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy.spatial.distance import euclidean
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'
for root, dirs, files in os.walk(base_path):
    sys.path.append(root)
import divisaodedados
import modelos
import shared_variables
#################################################################################

# Helper to get sorted unique peças
def get_sorted_pecas(peças):
    # Keep only unique, preserve order in peça_order, then add the rest
    peça_ordered = [p for p in peça_order if p in peças]
    rest = [p for p in peças if p not in peça_ordered]
    return peça_ordered + rest

# When creating legends, use sorted order
def sorted_legend_handles_labels(model_colors, present_pecas):
    sorted_pecas = get_sorted_pecas(present_pecas)
    handles = []
    labels = []
    for model in sorted_pecas:
        color = model_colors.get(model, '#333333')
        handles.append(plt.Rectangle((0,0),1,1, color=color))
        labels.append(model)
    return handles, labels

# Patch plot_angle_difference_bar to use sorted legend
def plot_angle_difference_bar(avg_non_compensated, avg_compensated, variable):
    psi_diff_non_comp = avg_non_compensated[variable] - new_data[variable].iloc[0]
    psi_diff_comp = avg_compensated[variable] - new_data[variable].iloc[0]

    plt.figure(figsize=(10, 6))
    for idx, peca in enumerate(avg_non_compensated.index):
        color = model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_non_comp[peca],
            color=color,
            alpha=0.7,
            label=f'Non-Compensated: {peca}' if idx == 0 else ""
        )
    for idx, peca in enumerate(avg_compensated.index):
        color = model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_comp[peca],
            color=color,
            alpha=0.4,
            label=f'Compensated: {peca}' if idx == 0 else ""
        )
    plt.xlabel('Peça')
    plt.ylabel(f'Difference in {latex_labels.get(variable, variable)}')
    plt.title(f'Difference in {latex_labels.get(variable, variable)} from New Data')
    # Sorted legend
    present_pecas = list(avg_non_compensated.index) + list(avg_compensated.index)
    handles, labels = sorted_legend_handles_labels(model_colors, present_pecas)
    plt.legend(handles, labels, title='Peça (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.tight_layout()
    plt.show()
peça_order = ['LRND', 'LRD', 'LRS', 'NN(8)', 'NN(16)']
# For avg_non_compensated and avg_compensated, reindex if all are present
def sort_peca_index(df):
    existing = [p for p in peça_order if p in df.index]
    rest = [p for p in df.index if p not in existing]
    return df.reindex(existing + rest)


def plot_angle_difference_bar(avg_non_compensated, avg_compensated, variable):
    """
    Plots a bar chart comparing the difference in angle (psi_D_deg) from new data
    for non-compensated and compensated datasets.

    Parameters:
        avg_non_compensated (pd.DataFrame): DataFrame with non-compensated averages, indexed by 'Peça'.
        avg_compensated (pd.DataFrame): DataFrame with compensated averages, indexed by 'Peça'.
        psi_diff_non_comp (pd.Series): Difference in angle for non-compensated.
        psi_diff_comp (pd.Series): Difference in angle for compensated.
    """
    # Calculate the difference between each average and the new data value
    psi_diff_non_comp = avg_non_compensated[variable] - new_data[variable].iloc[0]
    psi_diff_comp = avg_compensated[variable] - new_data[variable].iloc[0]


    plt.figure(figsize=(8, 6))
    handles = []
    labels = []
    # Plot non-compensated bars with model-specific colors
    for idx, peca in enumerate(avg_non_compensated.index):
        color = model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_non_comp[peca],
            color=color,
            label=f'Non-Compensated: {peca}' if idx == 0 else ""
        )
    # Plot compensated bars with model-specific colors (no hatch)
    for idx, peca in enumerate(avg_compensated.index):
        color = model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_comp[peca],
            color=color,
            label=f'Compensated: {peca}' if idx == 0 else ""
        )
    plt.xlabel('Peça', fontsize=16)
    plt.ylabel(f'Difference in {latex_labels.get(variable, variable)}', fontsize=16)
    plt.title(f'Difference in {latex_labels.get(variable, variable)} from New Data', fontsize=18)
    # Custom legend for models
    for model, color in model_colors.items():
        handles.append(plt.Rectangle((0,0),1,1, color=color))
        labels.append(model)
    # plt.legend(
    #     handles, labels, title='Peça (Model)',
    #     bbox_to_anchor=(0.5, -0.15), loc='upper center',
    #     ncol=max(1, len(handles)), fontsize=14, title_fontsize=15
    # )
    plt.grid(True)
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


angle_height_dataset_path = base_path + r'\csv_files\angle_height_dataset.csv'
compensated_angle_height_dataset_path = base_path + r'\csv_files\angle_height_compensated_parts.csv'
# Load the ensaios dataset
ensaios_path = base_path + r'\csv_files\ensaios.csv'



if os.path.exists(angle_height_dataset_path) and os.path.exists(compensated_angle_height_dataset_path):
    angle_height_dataset = pd.read_csv(angle_height_dataset_path).dropna()
    compensated_angle_height_dataset = pd.read_csv(compensated_angle_height_dataset_path).dropna()
    ensaios_dataset = pd.read_csv(ensaios_path)

    print(ensaios_dataset.head())
    print("Dataset loaded successfully:")
    print("Ensaios dataset loaded successfully:")
    print(angle_height_dataset.head())
    print(compensated_angle_height_dataset.head())
    print(ensaios_dataset)
else:
    print(f"One of the files does not exist.")





if (
not angle_height_dataset.empty
and not ensaios_dataset.empty
and 'Peça' in angle_height_dataset.columns
and 'Peça' in ensaios_dataset.columns
):
    combined_dataset = pd.merge(angle_height_dataset, ensaios_dataset, on='Peça', how='inner')
    print("Datasets merged successfully:")
    print(combined_dataset.head())

    # Step 4: Save the combined dataset
    combined_dataset_path = f"{base_path}\\csv_files\\combined_angle_height_ensaios.csv"
    combined_dataset.to_csv(combined_dataset_path, index=False)

else:
    print("Merging failed. Ensure both datasets have a 'Peça' column.")
    combined_dataset = pd.DataFrame()  # Define as empty to prevent further issues


# Separate data by unique pairs of ('psi_Cad', 'td', 'sd')
grouped = combined_dataset.groupby(['input angle', 'tool diameter', 'stepdown'])
compensated_grouped = compensated_angle_height_dataset.groupby(['Peça'])

# Remove rows where 'Peça' contains 'NN' but keep 'NN(8)' and 'NN(16)'
compensated_angle_height_dataset = compensated_angle_height_dataset[
    ~(
        compensated_angle_height_dataset['Peça'].str.contains('NN') &
        ~compensated_angle_height_dataset['Peça'].isin(['NN(8)', 'NN(16)'])
    )
]

# Sort the 'Peça' column so that 'NN(8)' appears before 'NN(16)'




# For compensated_angle_height_dataset, set categorical order if Peça exists
if 'Peça' in compensated_angle_height_dataset.columns:
    compensated_angle_height_dataset['Peça'] = pd.Categorical(
        compensated_angle_height_dataset['Peça'],
        categories=peça_order + [p for p in compensated_angle_height_dataset['Peça'].unique() if p not in peça_order],
        ordered=True
    )

# Access each group as a dictionary with tuple keys
grouped_data = {key: group for key, group in grouped}

# Print the unique pairs
print("Unique (psi_Cad, td, sd) pairs:", list(grouped_data.keys()))

# Access the group with psi_Cad=40, td=12, sd=0.7
group_40_12_07 = grouped_data.get((40, 12, 0.7))
print("Group for (psi_Cad=40, td=12, sd=0.7):")
print(group_40_12_07)

# Calculate average Angle (degrees) and Height for each unique Peça in group_40_12_07
if group_40_12_07 is not None:
    avg_non_compensated = group_40_12_07.groupby('Peça')[['Angle (degrees)', 'Height']].mean()
    print("Average values for non-compensated (group_40_12_07):")
    print(avg_non_compensated)
else:
    print("group_40_12_07 is None.")


# Update the 'Peça' values in compensated_angle_height_dataset
compensated_angle_height_dataset['Peça'] = compensated_angle_height_dataset['Peça'].replace({
    'LMA': 'LRND',
    'LMD': 'LRD',
    'LR': 'LRS'
})

angle_height_dataset['Peça'] = angle_height_dataset['Peça'].replace({
    'LMA': 'LRND',
    'LMD': 'LRD',
    'LR': 'LRS'
})

# Calculate average Angle (degrees) and Height for each unique Peça in compensated_grouped
avg_compensated = compensated_angle_height_dataset.groupby('Peça')[['Angle (degrees)', 'Height']].mean()


avg_non_compensated = sort_peca_index(avg_non_compensated)
avg_compensated = sort_peca_index(avg_compensated)

print("Average values for compensated:")
print(avg_compensated)



# Rename columns in avg_non_compensated and avg_compensated
avg_non_compensated = avg_non_compensated.rename(columns={
    'Angle (degrees)': 'psi_D_deg',
    'Height': 'h_D_mm',
    'Diameter': 'd_D_mm'
})
avg_compensated = avg_compensated.rename(columns={
    'Angle (degrees)': 'psi_D_deg',
    'Height': 'h_D_mm',
    'Diameter': 'd_D_mm'
})

# Also rename in the main datasets for consistency
angle_height_dataset = angle_height_dataset.rename(columns={
    'Angle (degrees)': 'psi_D_deg',
    'Height': 'h_D_mm',
    'stepdown': 'sd_mm',
    'tool diameter': 'd_t_mm',
    'Diameter': 'd_D_mm'
})
compensated_angle_height_dataset = compensated_angle_height_dataset.rename(columns={
    'Angle (degrees)': 'psi_D_deg',
    'Height': 'h_D_mm',
    'stepdown': 'sd_mm',
    'tool diameter': 'd_t_mm',
    'Diameter': 'd_D_mm'
})

new_data = pd.DataFrame([{           
        'psi_D_deg': 40,
        'h_D_mm': 25.17,
        'sd_mm': 0.7,
        'd_t_mm': 12,
        'd_D_mm': 20
    }])


latex_labels = {
    'psi_D_deg': r'$\Psi_m \, [^o]$',
    'h_D_mm': r'$h_m \, [mm]$',
    'diam_D_mm': r'$d_m \, [mm]$',
    'sd_mm': r'$sd \, [mm]$',
    'd_t_mm': r'$d_t \, [mm]$',
    'psi_CAD_deg': r'$\Psi_{CAD} \, [^o]$',
    'h_CAD_mm': r'$h_{CAD} \, [mm]$'
}




# Define a color mapping for each specific model (Peça)
model_colors = {
    'LRND': "#FF0000",
    'LRD': '#FF3399',
    'LRS': '#002EC0',
    'NN(8)': '#00C8BE',
    'NN(16)': '#2BA02B',
    'VII': "#454545",
    'III': "#454545",
    # Add more models and colors as needed
}



# Create a table with specific colors for each model
if 'Peça' in compensated_angle_height_dataset.columns:
    unique_models = compensated_angle_height_dataset['Peça'].unique()
    color_list = [model_colors.get(model, '#333333') for model in unique_models]
    table_data = []
    for model, color in zip(unique_models, color_list):
        table_data.append({
            'Model': model,
            'Color': color
        })
    color_table = pd.DataFrame(table_data)
    print("Model Color Table:")
    print(color_table)
else:
    print("Column 'Peça' not found in compensated_angle_height_dataset.")




# Plot the values of height and angle for each compensated part in the compensated_angle_height_dataset,
# coloring the points by the type of peça using the colors from the color table
if 'Peça' in compensated_angle_height_dataset.columns:
    plt.figure(figsize=(10, 6))
    pecas = compensated_angle_height_dataset['Peça'].unique()
    for peca in pecas:
        subset = compensated_angle_height_dataset[compensated_angle_height_dataset['Peça'] == peca]
        color = model_colors.get(peca, '#333333')  # Use color from model_colors, fallback to gray
        plt.scatter(
            subset['psi_D_deg'],
            subset['h_D_mm'],
            color=color,
            label=f'Peça: {peca}',
            s=80
        )
    # Add sorted legend
    present_pecas = list(pecas)
    handles, labels = sorted_legend_handles_labels(model_colors, present_pecas)
    plt.legend(handles, labels, title='Peça (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Add vertical and horizontal lines passing through new_data values
    plt.axvline(new_data['psi_D_deg'].iloc[0], color='gray', linestyle='--', linewidth=2, label='New Data psi_D_deg')
    plt.axhline(new_data['h_D_mm'].iloc[0], color='gray', linestyle='--', linewidth=2, label='New Data h_D_mm')
    plt.xlabel(latex_labels['psi_D_deg'])
    plt.ylabel(latex_labels['h_D_mm'])
    plt.title(f"{latex_labels['h_D_mm']} vs {latex_labels['psi_D_deg']} for Each Compensated Part (Colored by Peça)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Peça' not found in compensated_angle_height_dataset.")

# Calculate standard deviation of psi_D_deg and h_D_mm for each Peça in the compensated dataset
std_compensated = compensated_angle_height_dataset.groupby('Peça')[['psi_D_deg', 'h_D_mm']].std()
print("Standard deviation for each Peça in compensated dataset:")
print(std_compensated)

psi_diff_non_comp = avg_non_compensated['psi_D_deg'] - new_data['psi_D_deg'].iloc[0]
psi_diff_comp = avg_compensated['psi_D_deg'] - new_data['psi_D_deg'].iloc[0]
height_diff_non_comp = avg_non_compensated['h_D_mm'] - new_data['h_D_mm'].iloc[0]     
height_diff_comp = avg_compensated['h_D_mm'] - new_data['h_D_mm'].iloc[0]

print("psi_diff_non_comp:")
print(psi_diff_non_comp)
print("psi_diff_comp:")
print(psi_diff_comp)
print("height_diff_non_comp:")
print(height_diff_non_comp)
print("height_diff_comp:")
print(height_diff_comp)
plot_angle_difference_bar(avg_non_compensated, avg_compensated, 'psi_D_deg')
plot_angle_difference_bar(avg_non_compensated, avg_compensated, 'h_D_mm')

# Example usage:
# plot_angle_difference_bar(avg_non_compensated, avg_compensated, psi_diff_non_comp, psi_diff_comp)

# Calculate Euclidean distance between each average and the new data point

distances_non_comp = avg_non_compensated.apply(
    lambda row: euclidean([row['psi_D_deg'], row['h_D_mm']], [new_data['psi_D_deg'].iloc[0], new_data['h_D_mm'].iloc[0]]),
    axis=1
)
distances_comp = avg_compensated.apply(
    lambda row: euclidean([row['psi_D_deg'], row['h_D_mm']], [new_data['psi_D_deg'].iloc[0], new_data['h_D_mm'].iloc[0]]),
    axis=1
)


print("Euclidean distances (non-compensated):")
print(distances_non_comp)
print("Euclidean distances (compensated):")
print(distances_comp)
# Plot the differences with colors from the color table
plt.figure(figsize=(10, 6))

# Plot non-compensated points with a default color (since they may not have a model/color mapping)
for idx, (peca, x, y) in enumerate(zip(avg_non_compensated.index, psi_diff_non_comp, height_diff_non_comp)):
    color = model_colors.get(peca, '#333333')
    plt.scatter(
        x, y,
        color=color,
        label=f'Non-Compensated: {peca}' if idx == 0 else "",
        marker='o', s=80
    )

# Plot compensated points with their specific colors
for idx, (peca, x, y) in enumerate(zip(avg_compensated.index, psi_diff_comp, height_diff_comp)):
    color = model_colors.get(peca, '#333333')
    plt.scatter(
        x, y,
        color=color,
        label=f'Compensated: {peca}' if idx == 0 else "",
        marker='^', s=80
    )

# Add axial lines for new data psi and height
plt.axvline(0, color='black', linestyle='--', linewidth=2, label=f'{latex_labels["psi_D_deg"]} = 0')
plt.axhline(0, color='black', linestyle='--', linewidth=2, label=f'{latex_labels["h_D_mm"]} = 0')
plt.xlabel(f'Difference in {latex_labels["psi_D_deg"]}')
plt.ylabel(f'Difference in {latex_labels["h_D_mm"]}')
plt.title(f'Difference from New Data: Compensated vs Non-Compensated')

# Create custom legend for models
handles = []
labels = []
for model, color in model_colors.items():
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
    labels.append(model)
plt.legend(handles, labels, title='Peça (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.tight_layout()
plt.show()

# Merge angle_height_dataset with ensaios_dataset on 'Peça'
merged_dataset = pd.merge(angle_height_dataset, ensaios_dataset, on='Peça', how='inner')

# Boxplots for merged dataset parts with input angle of 40, stepdown 0.7, and tool diameter 12
filtered_merged = merged_dataset[
    (merged_dataset['input angle'] == 40) &
    (merged_dataset['stepdown'] == 0.7) &
    (merged_dataset['tool diameter'] == 12)
]

plt.figure(figsize=(8, 6))
sns.boxplot(
    data=compensated_angle_height_dataset,
    x='Peça',
    y='psi_D_deg',
    palette=model_colors
)
plt.xlabel('Peça', fontsize=16)
plt.ylabel(latex_labels.get('psi_D_deg', 'psi_D_deg'), fontsize=16)
plt.title(f'Boxplot of {latex_labels.get("psi_D_deg", "psi_D_deg")} for merged dataset (input angle={latex_labels.get("psi_CAD_deg", "input angle")}=40, {latex_labels.get("sd_mm", "stepdown")}=0.7, {latex_labels.get("d_t_mm", "tool diameter")}=12)', fontsize=18)
plt.grid(True, axis='y')
plt.hlines(
    y=new_data['psi_D_deg'].iloc[0],
    xmin=plt.gca().get_xlim()[0],
    xmax=plt.gca().get_xlim()[1],
    color='black',
    linestyle='--',
    linewidth=1,
    label=f'New Data {latex_labels.get("psi_D_deg", "psi_D_deg")}'
)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=compensated_angle_height_dataset,
    x='Peça',
    y='h_D_mm',
    palette=model_colors
)
plt.xlabel('Peça', fontsize=16)
plt.ylabel(latex_labels.get('h_D_mm', 'h_D_mm'), fontsize=16)
plt.title(f'Boxplot of {latex_labels.get("h_D_mm", "h_D_mm")} for merged dataset (input angle={latex_labels.get("psi_CAD_deg", "input angle")}=40, {latex_labels.get("sd_mm", "stepdown")}=0.7, {latex_labels.get("d_t_mm", "tool diameter")}=12)', fontsize=18)
plt.grid(True, axis='y')
plt.hlines(
    y=new_data['h_D_mm'].iloc[0],
    xmin=plt.gca().get_xlim()[0],
    xmax=plt.gca().get_xlim()[1],
    color='black',
    linestyle='--',
    linewidth=1,
    label=f'New Data {latex_labels.get("h_D_mm", "h_D_mm")}'
)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Create horizontal legend (labels in a single row) at the bottom
unique_models = compensated_angle_height_dataset['Peça'].unique()
handles = []
labels = []
for model in unique_models:
    color = model_colors.get(model, '#333333')
    handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10))
    labels.append(model)
# Make space at the bottom and place the legend in one row
plt.gcf().subplots_adjust(bottom=0.28)
# Order legend entries as: LRND, LRD, LRS, NN(8), NN(16) (display NN(16) as NN16)
preferred_order = ['LRND', 'LRD', 'LRS', 'NN(8)', 'NN(16)']
present_models = list(unique_models)

ordered_models = [m for m in preferred_order if m in present_models] + [m for m in present_models if m not in preferred_order]

handles = []
labels = []
for model in ordered_models:
    color = model_colors.get(model, '#333333')
    handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10))
    labels.append('NN16' if model == 'NN(16)' else model)
plt.legend(handles, labels, title='Peça (Model)', bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=max(1, len(handles)), fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.show()




# Create a color mapping for each unique (stepdown, tool diameter) pair
combo_cols = ['stepdown', 'tool diameter']
unique_pairs = merged_dataset[combo_cols].drop_duplicates()
pair_tuples = [tuple(row) for row in unique_pairs.values]
pair_colors = dict(zip(pair_tuples, sns.color_palette("husl", len(pair_tuples))))
from matplotlib.colors import to_hex
pair_colors = {k: to_hex(v) for k, v in pair_colors.items()}
ranges = [
    (37, 42),
    (48, 52),
    (58, 62),
    (68, 72)
]
marker_shape = 'o'  # Use a single marker shape for all points

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

for i, (psi_min, psi_max) in enumerate(ranges):
    ax = axes[i]
    filtered = merged_dataset[
        (merged_dataset['psi_D_deg'] >= psi_min) & 
        (merged_dataset['psi_D_deg'] <= psi_max)
    ]
    if not filtered.empty:
        avg_by_combo = (
            filtered
            .groupby(['stepdown', 'tool diameter'])[['psi_D_deg', 'h_D_mm']]
            .mean()
            .reset_index()
        )
        combo_cols = ['stepdown', 'tool diameter']
        unique_pairs = avg_by_combo[combo_cols].drop_duplicates()
        pair_tuples = [tuple(row) for row in unique_pairs.values]
        pair_colors = dict(zip(pair_tuples, sns.color_palette("husl", len(pair_tuples))))
        pair_colors = {k: to_hex(v) for k, v in pair_colors.items()}

        for _, row in avg_by_combo.iterrows():
            pair = (row['stepdown'], row['tool diameter'])
            color = pair_colors.get(pair, '#333333')
            ax.scatter(
                row['psi_D_deg'],
                row['h_D_mm'],
                color=color,
                marker=marker_shape,
                label=f"sd={row['stepdown']}, dt={row['tool diameter']}",
                s=80
            )
        # Custom legend for (stepdown, tool diameter) pairs
        handles = []
        labels = []
        for pair, color in pair_colors.items():
            handles.append(plt.Line2D([0], [0], marker=marker_shape, color='w', markerfacecolor=color, markersize=10))
            labels.append(f"sd={pair[0]}, dt={pair[1]}")
        ax.legend(handles, labels, title='Stepdown/Tool Diameter', bbox_to_anchor=(1.05, 1), loc='upper left')
        # Calculate and print the difference to the input angle value for each average

    else:
        ax.scatter(
            filtered['psi_D_deg'].mean(),
            filtered['h_D_mm'].mean(),
            color='#333333',
            s=80,
            alpha=0.7
        )
    ylim_set = False
    if 'input angle' in ensaios_dataset.columns and 'input height' in ensaios_dataset.columns:
        ensaios_in_range = ensaios_dataset[
            (ensaios_dataset['input angle'] >= psi_min) & 
            (ensaios_dataset['input angle'] <= psi_max)
        ]
        for idx, row in ensaios_in_range.iterrows():
            ax.axvline(row['input angle'], color='gray', linestyle='--', linewidth=1)
            ax.axhline(row['input height'], color='gray', linestyle='--', linewidth=1)
        if not ensaios_in_range.empty:
            ensaios_height = ensaios_in_range['input height'].iloc[0]
            ensaios_angle = ensaios_in_range['input angle'].iloc[0]
            ax.set_ylim(ensaios_height - 1, ensaios_height + 1)
            ax.set_xlim(ensaios_angle - 1.5, ensaios_angle + 1.5)
            ylim_set = True
            xlim_set = True


    ax.set_xlabel(latex_labels.get('psi_D_deg', 'psi_D_deg'))
    ax.set_ylabel(latex_labels.get('h_D_mm', 'h_D_mm'))
    ax.set_title(f'Angle vs Height for psi_D_deg in [{psi_min}, {psi_max}]')
    ax.grid(True)
    
    diff = {}
    avg_by_combo['h_D_mm_diff'] = avg_by_combo['h_D_mm'] - ensaios_height
    avg_by_combo['psi_D_deg_diff'] = avg_by_combo['psi_D_deg'] - ensaios_angle
    print(f"Range psi_D_deg in [{psi_min}, {psi_max}]:")
    
    print(avg_by_combo)
plt.tight_layout()
plt.show()
