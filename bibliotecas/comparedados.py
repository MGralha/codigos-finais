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


# Helper to get sorted unique peças
def get_sorted_pecas(peças):
    # Keep only unique, preserve order in shared_variables.peça_order, then add the rest
    shared_variables.peça_ordered = [p for p in shared_variables.peça_order if p in peças]
    rest = [p for p in peças if p not in shared_variables.peça_ordered]
    return shared_variables.peça_ordered + rest

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
    psi_diff_non_comp = avg_non_compensated[variable] - shared_variables.new_data[variable].iloc[0]
    psi_diff_comp = avg_compensated[variable] - shared_variables.new_data[variable].iloc[0]

    plt.figure(figsize=(10, 6))
    for idx, peca in enumerate(avg_non_compensated.index):
        color = shared_variables.model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_non_comp[peca],
            color=color,
            alpha=0.7,
            label=f'Non-Compensated: {peca}' if idx == 0 else ""
        )
    for idx, peca in enumerate(avg_compensated.index):
        color = shared_variables.model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_comp[peca],
            color=color,
            alpha=0.4,
            label=f'Compensated: {peca}' if idx == 0 else ""
        )
    plt.xlabel('Peça')
    plt.ylabel(f'Difference in {shared_variables.latex_labels.get(variable, variable)}')
    plt.title(f'Difference in {shared_variables.latex_labels.get(variable, variable)} from New Data')
    # Sorted legend
    present_pecas = list(avg_non_compensated.index) + list(avg_compensated.index)
    handles, labels = sorted_legend_handles_labels(shared_variables.model_colors, present_pecas)
    plt.legend(handles, labels, title='Peça (Model)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.tight_layout()
    plt.show()

# For avg_non_compensated and avg_compensated, reindex if all are present
def sort_peca_index(df):
    existing = [p for p in shared_variables.peça_order if p in df.index]
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
    psi_diff_non_comp = avg_non_compensated[variable] - shared_variables.new_data[variable].iloc[0]
    psi_diff_comp = avg_compensated[variable] - shared_variables.new_data[variable].iloc[0]


    plt.figure(figsize=(8, 6))
    handles = []
    labels = []
    # Plot non-compensated bars with model-specific colors
    for idx, peca in enumerate(avg_non_compensated.index):
        color = shared_variables.model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_non_comp[peca],
            color=color,
            label=f'Non-Compensated: {peca}' if idx == 0 else ""
        )
    # Plot compensated bars with model-specific colors (no hatch)
    for idx, peca in enumerate(avg_compensated.index):
        color = shared_variables.model_colors.get(peca, '#333333')
        plt.bar(
            peca,
            psi_diff_comp[peca],
            color=color,
            label=f'Compensated: {peca}' if idx == 0 else ""
        )
    plt.xlabel('Peça', fontsize=16)
    plt.ylabel(f'Difference in {shared_variables.latex_labels.get(variable, variable)}', fontsize=16)
    plt.title(f'Difference in {shared_variables.latex_labels.get(variable, variable)} from New Data', fontsize=18)
    # Custom legend for models
    for model, color in shared_variables.model_colors.items():
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

