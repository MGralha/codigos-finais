import pandas as pd
import os
import numpy as np
from scipy import optimize as opt
from scipy.stats import gamma
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from sklearn.neural_network import MLPRegressor
from scipy.stats import shapiro
from sklearn.feature_selection import mutual_info_regression
import dcor
from sklearn.model_selection import KFold
import matplotlib
import builtins
import shared_variables

def store_predictions_table(prediction_table, fold_idx, model_name, target_names, Y_true, Y_pred, indices):
    """
    Store predicted values and true values for each model in a table.
    Appends to prediction_table a dict for each prediction with columns:
    Fold, Model, Target, Index, TrueValue, PredictedValue
    """
    for i, target in enumerate(target_names):
        for idx, true_val, pred_val in zip(indices, Y_true.iloc[:, i], Y_pred[:, i] if Y_pred.ndim > 1 else Y_pred):
            prediction_table.append({
                'Fold': fold_idx + 1,
                'Model': model_name,
                'Target': target,
                'Index': idx,
                'TrueValue': true_val,
                'PredictedValue': pred_val
            })
    return prediction_table

def split_data(X, Y, method='kfold', n_splits=5, test_size=0.2, random_state=42, group_keys=None):
    """
    Split data using either k-fold or random train/test split.
    - method: 'kfold' or 'random'
    - n_splits: number of folds for kfold
    - test_size: test size for random split
    - group_keys: Series or array for group-wise kfold (tuple of columns)
    Returns: X_train, X_test, Y_train, Y_test
    """
    if method == 'kfold':
        if group_keys is None:
            raise ValueError("group_keys must be provided for kfold splitting.")
        data = X.copy()
        data['group_key'] = group_keys
        group_folds = {}
        for group, group_df in data.groupby('group_key'):
            kf = KFold(n_splits=min(n_splits, len(group_df)), shuffle=True, random_state=random_state)
            group_folds[group] = []
            for _, test_idx in kf.split(group_df):
                group_folds[group].append(group_df.index[test_idx])
        return group_folds
    elif method == 'random':
        return train_test_split(X, Y, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("method must be 'kfold' or 'random'.")


def prepare_group_data(df, group, rename_dict, cols_to_drop):
    group_df = df[df['Peça'].isin(group)].drop(columns=cols_to_drop, errors='ignore')
    group_df = group_df.rename(columns=rename_dict)
    group_df = group_df.apply(pd.to_numeric, errors='coerce')
    return group_df

def simulate_correlated_mixture(n_samples=1000, means=[(0, 0), (3, 3)], covs=[[[1, 0.8], [0.8, 1]], [[1, -0.8], [-0.8, 1]]], weights=[0.5, 0.5], random_state=42):
    """
    Simulate a mixture of two bivariate normal distributions with specified means, covariances, and weights.
    Returns a DataFrame with columns ['X', 'Y', 'Component'].
    """
    np.random.seed(random_state)
    n1 = int(n_samples * weights[0])
    n2 = n_samples - n1
    X1 = np.random.multivariate_normal(means[0], covs[0], n1)
    X2 = np.random.multivariate_normal(means[1], covs[1], n2)
    df1 = pd.DataFrame(X1, columns=['X', 'Y'])
    df1['Component'] = 0
    df2 = pd.DataFrame(X2, columns=['X', 'Y'])
    df2['Component'] = 1
    df = pd.concat([df1, df2], ignore_index=True)
    return df


def plot_correlation_heatmap(data, latex_labels, title):
    corr = data.corr(method='pearson')
    corr_display = corr.rename(index=latex_labels, columns=latex_labels)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor='white',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot_kws={"fontsize": 14}
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# Function to calculate and plot distance correlation heatmap
def plot_distance_correlation_heatmap(data, latex_labels, title):
    dcorr_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            dcorr_matrix[i, j] = dcor.distance_correlation(data.iloc[:, i], data.iloc[:, j])

    dcorr_df = pd.DataFrame(dcorr_matrix, index=data.columns, columns=data.columns)
    dcorr_display = dcorr_df.rename(index=latex_labels, columns=latex_labels)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        dcorr_display,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        cbar_kws={"shrink": 0.8, "ticks": np.linspace(-1, 1, 5)},
        linewidths=0.5,
        linecolor='white',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot_kws={"fontsize": 14}
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()





def plot_spearman_heatmap(data, latex_labels, title):
            corr = data.corr(method='spearman')
            corr_display = corr.rename(index=latex_labels, columns=latex_labels)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
            corr_display,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
            linecolor='white',
            vmin=-1,
            vmax=1,
            xticklabels=True,
            yticklabels=True,
            annot_kws={"fontsize": 14}
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
            plt.title(title, fontsize=16)
            plt.tight_layout()
            plt.show()
    
def plot_kendall_heatmap(data, latex_labels, title):
            corr = data.corr(method='kendall')
            corr_display = corr.rename(index=latex_labels, columns=latex_labels)
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                corr_display,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                square=True,
                cbar_kws={"shrink": 0.8},
                linewidths=0.5,
                linecolor='white',
                vmin=-1,
                vmax=1,
                xticklabels=True,
                yticklabels=True,
                annot_kws={"fontsize": 14}
            )
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45, ha='right')
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
            plt.title(title, fontsize=16)
            plt.tight_layout()
            plt.show()

# Function to plot a polychromatic (multi-color) correlation heatmap
def plot_polychrolic_heatmap(data, latex_labels, title):
    corr = data.corr(method='pearson')
    corr_display = corr.rename(index=latex_labels, columns=latex_labels)
    plt.figure(figsize=(10, 8))
    # Use a polychromatic colormap, e.g., 'Spectral' or 'nipy_spectral'
    ax = sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        linecolor='white',
        vmin=-1,
        vmax=1,
        xticklabels=True,
        yticklabels=True,
        annot_kws={"fontsize": 14}
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16, rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16, rotation=0)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def correlation_and_mi_analysis(combined_dataset, group40_data, group70_data, group50_data, group60_data, latex_labels):
    # Define input/output features
    input_features = ['psi_D_deg', 'h_D_mm', 'sd_mm', 'd_t_mm']
    output_targets = ['psi_CAD_deg', 'h_CAD_mm']

    # Drop rows with missing values for these features
    data = combined_dataset.dropna(subset=input_features + output_targets)
    X = data[input_features]

    # Mutual Information analysis for input features vs each output
    print("\n--- Mutual Information Dependence Study ---")
    for target in output_targets:
        mi = mutual_info_regression(X, data[target], random_state=42)
        print(f"\nMutual Information between input features and '{target}':")
        for feature, score in zip(input_features, mi):
            print(f"  {feature} -> {target}: {score:.4f}")

    # Bar plot for visualization
    plt.figure(figsize=(6, 4))
    plt.bar(input_features, mi, color='teal')
    plt.xlabel('Input Feature')
    plt.ylabel('Mutual Information')
    plt.title(f"Mutual Information: Inputs vs {target}")
    plt.tight_layout()
    plt.show()

    # --- Simulation of Correlated Mixture Distributions ---
    mixture_df = simulate_correlated_mixture(n_samples=2000)

    # Plot the simulated mixture
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=mixture_df, x='X', y='Y', hue='Component', palette='Set1', alpha=0.5)
    plt.title('Simulated Correlated Mixture Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Compute and print correlations for the whole mixture and each component
    print("Correlation (Pearson) for full mixture:", mixture_df[['X', 'Y']].corr().iloc[0, 1])
    for comp in [0, 1]:
        comp_corr = mixture_df[mixture_df['Component'] == comp][['X', 'Y']].corr().iloc[0, 1]
        print(f"Correlation (Pearson) for component {comp}:", comp_corr)

    # Distance correlation for the mixture and each component
    print("Distance correlation (full mixture):", dcor.distance_correlation(mixture_df['X'], mixture_df['Y']))
    for comp in [0, 1]:
        dcorr = dcor.distance_correlation(
            mixture_df[mixture_df['Component'] == comp]['X'],
            mixture_df[mixture_df['Component'] == comp]['Y']
        )
    print(f"Distance correlation (component {comp}):", dcorr)

    

    # Drop output columns for correlation
    group40_corr = group40_data.drop(columns=['psi_CAD_deg', 'h_CAD_mm'], errors='ignore')
    group70_corr = group70_data.drop(columns=['psi_CAD_deg', 'h_CAD_mm'], errors='ignore')
    group50_corr = group50_data.drop(columns=['psi_CAD_deg', 'h_CAD_mm'], errors='ignore')
    group60_corr = group60_data.drop(columns=['psi_CAD_deg', 'h_CAD_mm'], errors='ignore')

    # Plot histograms for psi_D_deg and h_D_mm for each group
    for group_name, group_df in zip(
    ['Group 40', 'Group 70', 'Group 50', 'Group 60'],
    [group40_corr, group70_corr, group50_corr, group60_corr]
    ):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        # Define bin edges with interval 0.1 over the data range
        psi_min = group_df['psi_D_deg'].min()
        psi_max = group_df['psi_D_deg'].max()
        psi_avg = group_df['psi_D_deg'].mean()
        bins = np.arange(psi_min, psi_max + 0.05, 0.05)
        plt.hist(group_df['psi_D_deg'].dropna(), bins=bins, color='skyblue', edgecolor='black')
        # Set x-axis ticks every 3 degrees
        # Set x-axis limits to be 2 units from the lowest and highest bar of the histogram
        plt.xlim(psi_avg - 0.75, psi_avg + 0.75)
        # Set x-ticks every 0.1, starting at the next lowest even number
        start_tick = np.floor(psi_avg - 0.75)
        end_tick = np.ceil(psi_avg + 0.75)
        plt.xticks(np.arange(start_tick, end_tick + 0.25, 0.25))
        plt.xlabel(r'$\Psi_D \, [^o]$', fontsize=14)

        plt.ylabel('Frequency')
        plt.title(f'{group_name}: Histogram of psi_D_deg')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        h_min = group_df['h_D_mm'].min()
        h_max = group_df['h_D_mm'].max()
        h_avg = group_df['h_D_mm'].mean()
        bins = np.arange(h_min, h_max + 0.1, 0.1)
        plt.hist(group_df['h_D_mm'].dropna(), bins=bins, color='salmon', edgecolor='black')
        plt.xlim(h_avg - 0.75, h_avg + 0.75)
        start_tick = np.floor(h_avg - 0.75)
        end_tick = np.ceil(h_avg + 0.75)
        plt.xticks(np.arange(start_tick, end_tick + 0.25, 0.25))
        plt.xlabel(r'$h_D \, [mm]$', fontsize=14)
        plt.ylabel('Frequency')
        plt.title(f'{group_name}: Histogram of h_D_mm')
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()

    # Plot distance correlation heatmaps
    plot_distance_correlation_heatmap(group40_corr, latex_labels, "Distance Correlation Heatmap - Group 40")
    plot_distance_correlation_heatmap(group70_corr, latex_labels, "Distance Correlation Heatmap - Group 70")
    plot_distance_correlation_heatmap(group50_corr, latex_labels, "Distance Correlation Heatmap - Group 50")
    plot_distance_correlation_heatmap(group60_corr, latex_labels, "Distance Correlation Heatmap - Group 60")
    plot_distance_correlation_heatmap(combined_dataset.select_dtypes(include=np.number), latex_labels, "Distance Correlation Heatmap")

    # Plot Pearson heatmaps
    plot_correlation_heatmap(group40_corr, latex_labels, "Pearson Correlation Heatmap - Group 40")
    plot_correlation_heatmap(group70_corr, latex_labels, "Pearson Correlation Heatmap - Group 70")
    plot_correlation_heatmap(group50_corr, latex_labels, "Pearson Correlation Heatmap - Group 50")
    plot_correlation_heatmap(group60_corr, latex_labels, "Pearson Correlation Heatmap - Group 60")
    plot_correlation_heatmap(combined_dataset, latex_labels, "Pearson Correlation Heatmap")

    # --- Spearman correlation analysis ---
    plot_spearman_heatmap(group40_corr, latex_labels, "Spearman Correlation Heatmap - Group 40")
    plot_spearman_heatmap(group70_corr, latex_labels, "Spearman Correlation Heatmap - Group 70")
    plot_spearman_heatmap(group50_corr, latex_labels, "Spearman Correlation Heatmap - Group 50")
    plot_spearman_heatmap(group60_corr, latex_labels, "Spearman Correlation Heatmap - Group 60")
    plot_spearman_heatmap(combined_dataset, latex_labels, "Spearman Correlation Heatmap")

    # --- Kendall Tau correlation analysis ---
    plot_kendall_heatmap(group40_corr, latex_labels, "Kendall Tau Correlation Heatmap - Group 40")
    plot_kendall_heatmap(group70_corr, latex_labels, "Kendall Tau Correlation Heatmap - Group 70")
    plot_kendall_heatmap(group50_corr, latex_labels, "Kendall Tau Correlation Heatmap - Group 50")
    plot_kendall_heatmap(group60_corr, latex_labels, "Kendall Tau Correlation Heatmap - Group 60")
    plot_kendall_heatmap(combined_dataset, latex_labels, "Kendall Tau Correlation Heatmap")



    # Example usage for each group and combined dataset
    plot_polychrolic_heatmap(group40_corr, latex_labels, "Polychrolic Correlation Heatmap - Group 40")
    plot_polychrolic_heatmap(group70_corr, latex_labels, "Polychrolic Correlation Heatmap - Group 70")
    plot_polychrolic_heatmap(group50_corr, latex_labels, "Polychrolic Correlation Heatmap - Group 50")
    plot_polychrolic_heatmap(group60_corr, latex_labels, "Polychrolic Correlation Heatmap - Group 60")
    plot_polychrolic_heatmap(combined_dataset.select_dtypes(include=np.number), latex_labels, "Polychrolic Correlation Heatmap")



# Define the function for linear regression and 3D plots
def linear_regression_and_3d_plots(X_train, X_test, Y_train, Y_test, output_targets, group_40_mask, group_70_mask, group_50_mask, group_60_mask, latex_labels, new_data, prediction_table, fold_idx):
    model = LinearRegression()
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # Print regression equations
    print("Intercepts:", model.intercept_)
    for i, output_name in enumerate(Y_train.columns):
        print(f"\nRegression equation for '{output_name}':")
        print(f"{output_name} = {model.intercept_[i]:.4f}", end='')
        for coef, feature in zip(model.coef_[i], X_train.columns):
            print(f" + ({coef:.4f} * {feature})", end='')
        print()

    # Evaluation


    # 3D plots for each group
    for i, target in enumerate(output_targets):
        for group_label, group_mask, color_map in [
            ('Group 40 (input angle = 40)', group_40_mask, 'coolwarm'),
            ('Group 70 (input angle = 70)', group_70_mask, 'coolwarm'),
            ('Group 50 (input angle = 50)', group_50_mask, 'coolwarm'),
            ('Group 60 (input angle = 60)', group_60_mask, 'coolwarm')
        ]:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            group_indices = Y_test.index[group_mask]
            X_group = X_test.loc[group_indices]
            Y_actual_group = Y_test.loc[group_indices, target]
            Y_pred_group = Y_pred[group_mask, i]

            scatter = ax.scatter(
                X_group['psi_D_deg'],
                X_group['h_D_mm'],
                Y_pred_group,
                c=Y_actual_group - Y_pred_group,
                cmap=color_map,
                s=50,
                label=group_label,
                vmin=-0.5,
                vmax=0.5
            )

            angle_range = np.linspace(X_group['psi_D_deg'].min(), X_group['psi_D_deg'].max(), 30)
            height_range = np.linspace(X_group['h_D_mm'].min(), X_group['h_D_mm'].max(), 30)
            a, h = np.meshgrid(angle_range, height_range)

            fixed_stepdown = X_group['sd_mm'].mean()
            fixed_diameter = X_group['d_t_mm'].mean()

            model_input = pd.DataFrame({
                'psi_D_deg': a.ravel(),
                'h_D_mm': h.ravel(),
                'sd_mm': np.full(a.size, fixed_stepdown),
                'd_t_mm': np.full(a.size, fixed_diameter)
            })

            Z = model.predict(model_input)[:, i].reshape(a.shape)
            ax.plot_surface(a, h, Z, alpha=0.3, cmap='grey', edgecolor='none')
            ax.set_xlabel(latex_labels['psi_D_deg'], fontsize=16)
            ax.set_ylabel(latex_labels['h_D_mm'], fontsize=16)
            ax.set_zlabel(f'Predicted {latex_labels[target]}', fontsize=16)
            ax.set_title(f"3D Scatter + Regression Surface\n{latex_labels[target]} - {group_label}")
            plt.colorbar(scatter, label='Actual - Predicted')
            plt.tight_layout()
            plt.legend()
            plt.show()

    # All-data plot
    for i, target in enumerate(output_targets):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        X_all = X_test
        Y_actual_all = Y_test.iloc[:, i]
        Y_pred_all = Y_pred[:, i]

        scatter = ax.scatter(
            X_all['psi_D_deg'],
            X_all['h_D_mm'],
            Y_pred_all,
            c=Y_actual_all - Y_pred_all,
            cmap='coolwarm',
            s=50,
            label='All Data',
            vmin=-0.5,
            vmax=0.5
        )

        angle_range = np.linspace(X_all['psi_D_deg'].min(), X_all['psi_D_deg'].max(), 30)
        height_range = np.linspace(X_all['h_D_mm'].min(), X_all['h_D_mm'].max(), 30)
        a, h = np.meshgrid(angle_range, height_range)

        fixed_stepdown = X_all['sd_mm'].mean()
        fixed_diameter = X_all['d_t_mm'].mean()

        model_input = pd.DataFrame({
            'psi_D_deg': a.ravel(),
            'h_D_mm': h.ravel(),
            'sd_mm': np.full(a.size, fixed_stepdown),
            'd_t_mm': np.full(a.size, fixed_diameter)
        })

        Z = model.predict(model_input)[:, i].reshape(a.shape)

        ax.plot_surface(a, h, Z, alpha=0.3, color='grey', edgecolor='none')
        ax.set_xlabel(latex_labels['psi_D_deg'], fontsize=16)
        ax.set_ylabel(latex_labels['h_D_mm'], fontsize=16)
        ax.set_zlabel(f'Predicted {latex_labels[target]}', fontsize=16)
        ax.set_title(f"3D Scatter + Regression Surface for {latex_labels[target]} (All Data)")
        plt.colorbar(scatter, label='Actual - Predicted')
        plt.tight_layout()
        plt.legend()
        plt.show()
        # Predict for new data using the trained linear regression model
        predicted_output = model.predict(new_data)
        print(f"Predicted input angle: {predicted_output[0][0]:.2f}")
        print(f"Predicted input height: {predicted_output[0][1]:.2f}")

        # Store the predicted output for new_data in the prediction_table
        prediction_table = store_predictions_table(
            prediction_table,
            fold_idx,
            'Linear Regression',
            output_targets,
            new_data,  # True values are not available for new_data, so we can store NaN or the input itself
            predicted_output,
            new_data.index
        )

        # Evaluation for each output target
    r2_scores = []
    mses = []
    for i, target in enumerate(output_targets):
        r2 = r2_score(Y_test.iloc[:, i], Y_pred[:, i])
        mse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i])
        r2_scores.append(r2)
        mses.append(mse)
        print(f"\n--- {target} ---")
        print("R^2 score:", r2)
        print("MSE:", mse)
    # Return R2 and MSE for both outputs for easier table creation
    return model, Y_pred, r2_scores, mses



def plot_3d_actual_vs_predicted(X_test, Y_test, Y_pred, output_targets, group_masks, latex_labels, model):
    """
    Plots 3D scatter and regression surface for actual vs predicted outputs.
    group_masks: list of (group_label, mask, color_map)
    """
    for i, target in enumerate(output_targets):
        for group_label, group_mask, color_map in group_masks:
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')

            group_indices = Y_test.index[group_mask]
            X_group = X_test.loc[group_indices]
            Y_actual_group = Y_test.loc[group_indices, target]
            Y_pred_group = Y_pred[group_mask, i]

            # scatter = ax.scatter(
            #     X_group['psi_D_deg'],
            #     X_group['h_D_mm'],
            #     Y_actual_group,
            #     c=Y_actual_group - Y_pred_group,
            #     cmap=color_map,
            #     s=50,
            #     label=group_label,
            #     vmin=-0.5,
            #     vmax=0.5
            # )

            angle_range = np.linspace(X_group['psi_D_deg'].min(), X_group['psi_D_deg'].max(), 30)
            height_range = np.linspace(X_group['h_D_mm'].min(), X_group['h_D_mm'].max(), 30)
            a, h = np.meshgrid(angle_range, height_range)

            fixed_stepdown = X_group['sd_mm'].mean()
            fixed_diameter = X_group['d_t_mm'].mean()

            model_input = pd.DataFrame({
                'psi_D_deg': a.ravel(),
                'h_D_mm': h.ravel(),
                'sd_mm': np.full(a.size, fixed_stepdown),
                'd_t_mm': np.full(a.size, fixed_diameter)
            })

            Z_surface = model.predict(model_input)[:, i].reshape(a.shape)

            # ax.plot_surface(a, h, Z_surface, alpha=0.3, cmap=color_map, edgecolor='none')
            # ax.set_xlabel(latex_labels['psi_D_deg'])
            # ax.set_ylabel(latex_labels['h_D_mm'])
            # ax.set_zlabel(latex_labels[target])
            # ax.set_title(f"3D Scatter + Regression Surface\n{latex_labels[target]} - {group_label}")
            # plt.colorbar(scatter, label='Actual - Predicted')
            # plt.tight_layout()
            # plt.legend()
            # plt.show()

    for i, target in enumerate(output_targets):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        X_all = X_test
        Y_actual_all = Y_test.iloc[:, i]
        Y_pred_all = Y_pred[:, i]

        # scatter = ax.scatter(
        #     X_all['psi_D_deg'],
        #     X_all['h_D_mm'],
        #     Y_actual_all,
        #     c=Y_actual_all - Y_pred_all,
        #     cmap='coolwarm',
        #     s=50,
        #     label='All Data',
        #     vmin=-0.5,
        #     vmax=0.5
        # )

        angle_range = np.linspace(X_all['psi_D_deg'].min(), X_all['psi_D_deg'].max(), 30)
        height_range = np.linspace(X_all['h_D_mm'].min(), X_all['h_D_mm'].max(), 30)
        a, h = np.meshgrid(angle_range, height_range)

        fixed_stepdown = X_all['sd_mm'].mean()
        fixed_diameter = X_all['d_t_mm'].mean()

        model_input = pd.DataFrame({
            'psi_D_deg': a.ravel(),
            'h_D_mm': h.ravel(),
            'sd_mm': np.full(a.size, fixed_stepdown),
            'd_t_mm': np.full(a.size, fixed_diameter)
        })

        Z_surface = model.predict(model_input)[:, i].reshape(a.shape)

        # ax.plot_surface(a, h, Z_surface, alpha=0.3, cmap='coolwarm', edgecolor='none')
        # ax.set_xlabel(latex_labels['psi_D_deg'])
        # ax.set_ylabel(latex_labels['h_D_mm'])
        # ax.set_zlabel(latex_labels[target])
        # ax.set_title(f"3D Scatter + Regression Surface for {latex_labels[target]} (All Data)")
        # plt.colorbar(scatter, label='Actual - Predicted')
        # plt.tight_layout()
        # plt.legend()
        # plt.show()


def fit_and_plot_tangent(all_angles, all_heights, X_test_angle, Y_test_height):
    """
    Fit a tangent function to the given angles and heights, plot the fit, and return fit parameters.
    """
    p0 = [30, 1, 0, 0]  # Initial guess: a=30, b=1, c=0, d=0
    try:
        popt, _ = opt.curve_fit(tangent_func, all_angles, all_heights, p0=p0, maxfev=10000)
        fit_label = f'Best Fit: {popt[0]:.2f}·tan({popt[1]:.2f}·Angle + {popt[2]:.2f}) + {popt[3]:.2f}'
        angle_line = np.linspace(35, 75, 100)
        fit_line = tangent_func(angle_line, *popt)
        # plt.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)

        if popt is not None:
            Y_pred_test = tangent_func(X_test_angle, *popt)
            r2 = r2_score(Y_test_height, Y_pred_test)
            mse = mean_squared_error(Y_test_height, Y_pred_test)
            print(f"R^2 score for tangent model (test data): {r2}")
            print("MSE:", mse)
            return popt, fit_line, angle_line, fit_label, r2, mse
        else:
            return popt, fit_line, angle_line, fit_label, None, None

    except Exception as e:
        print(f"Tangent fit failed: {e}")
        return None, None, None, None


def tangent_func(x, a, b, c, d):
    return a * np.tan(b * np.radians(x) + c) + d

def plot_linear_regression_all(X_train_angle, Y_train_CAD_angle, all_angles, CAD_all_angles,
                    X_train_height, Y_train_CAD_height, all_heights, CAD_all_heights, X_test_angle, Y_test_angle, X_test_height, Y_test_height):
    # Linear regression for angle
    model_all = LinearRegression()
    model_all.fit(X_train_angle.reshape(-1, 1), Y_train_CAD_angle)
    slope_all = model_all.coef_[0]
    intercept_all = model_all.intercept_

    angle_line_all = np.linspace(20, 90, 200)
    linear_fit_all = slope_all * angle_line_all + intercept_all
    # plt.figure(figsize=(8, 6))
    # plt.plot(angle_line_all, linear_fit_all, label=f'All Data: $\\psi_{{CAD}} = {slope_all:.2f} \\, \\psi_D + {intercept_all:.2f}$', color='magenta')
    # plt.scatter(all_angles, CAD_all_angles, color='blue', alpha=0.5, label='All Data Points')
    # plt.xlabel('Measured Angle [º]', fontsize=14)
    # plt.ylabel('Input Angle (CAD) [º]', fontsize=14)
    # plt.title('Linear Regression: CAD Angle vs Measured Angle (All Data)')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()


    # Evaluate linear model for angle (psi_CAD_deg)
    Y_pred_linear_angle = model_all.predict(X_test_angle.reshape(-1, 1))
    print(f"Linear Model (Angle): R^2 = {r2_score(Y_test_angle, Y_pred_linear_angle):.4f}")
    print(f"Linear Model (Angle): MSE = {mean_squared_error(Y_test_angle, Y_pred_linear_angle):.4f}")

    # Linear regression for height
    model_all_height = LinearRegression()
    model_all_height.fit(X_train_height.reshape(-1, 1), Y_train_CAD_height)
    slope_all_height = model_all_height.coef_[0]
    intercept_all_height = model_all_height.intercept_

    height_line_all = np.linspace(20, 90, 200)
    linear_fit_all_height = slope_all_height * height_line_all + intercept_all_height
    # plt.figure(figsize=(8, 6))
    # plt.plot(height_line_all, linear_fit_all_height, label=f'All Data: $h_{{CAD}} = {slope_all_height:.2f} \\, h_D + {intercept_all_height:.2f}$', color='magenta')
    # plt.scatter(all_heights, CAD_all_heights, color='green', alpha=0.5, label='All Data Points')
    # plt.xlabel('Measured Height [mm]', fontsize=14)
    # plt.ylabel('Input Height (CAD) [mm]', fontsize=14)
    # plt.title('Linear Regression: CAD Height vs Measured Height (All Data)')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()

    
    # Evaluate linear model for angle (psi_CAD_deg)
    Y_pred_linear_height = model_all_height.predict(X_test_height.reshape(-1, 1))
    print(f"Linear Model (Height): R^2 = {r2_score(Y_test_height, Y_pred_linear_height):.4f}")
    print(f"Linear Model (Height): MSE = {mean_squared_error(Y_test_height, Y_pred_linear_height):.4f}")

    return model_all, model_all_height

# Prepare train data for each group (by psi_CAD_deg)
def get_group(X_set, Y_set, angle):
    mask = Y_set['psi_CAD_deg'] == angle
    return X_set.loc[mask], Y_set.loc[mask]

def analyze_by_sd_td(
    unique_sd, unique_td,
    group40_data, group70_data, group50_data, group60_data,
    X_train_40, Y_train_40, X_train_50, Y_train_50, X_train_60, Y_train_60, X_train_70, Y_train_70,
    X_test_40, Y_test_40, X_test_50, Y_test_50, X_test_60, Y_test_60, X_test_70, Y_test_70,
    X_test, Y_test,
    all_angles, CAD_all_angles, all_heights, CAD_all_heights,
    fit_and_plot_tangent, plot_linear_regression_all, plot_group_scatter_and_subplots
):
    fitted_params = []
    linear_params_h = []
    linear_params_angle = []
    for sd in unique_sd:
        for td in unique_td:
             # Get group data for this sd and td
            g40 = group40_data[(group40_data['sd_mm'] == sd) & (group40_data['d_t_mm'] == td)]
            g70 = group70_data[(group70_data['sd_mm'] == sd) & (group70_data['d_t_mm'] == td)]
            g50 = group50_data[(group50_data['sd_mm'] == sd) & (group50_data['d_t_mm'] == td)]
            g60 = group60_data[(group60_data['sd_mm'] == sd) & (group60_data['d_t_mm'] == td)]

            X_test_sdtd = X_test[(X_test['sd_mm'] == sd) & (X_test['d_t_mm'] == td)]
            Y_test_sdtd = Y_test.loc[X_test_sdtd.index]
            X_test_height = X_test_sdtd['h_D_mm'].values
            X_test_angle = X_test_sdtd['psi_D_deg'].values
            Y_test_height = Y_test_sdtd['h_CAD_mm'].values
            Y_test_angle = Y_test_sdtd['psi_CAD_deg'].values    

            # Only plot if at least one group has data
            if any([len(x) > 0 for x in [all_angles, all_heights]]):
                # --- Use tangent function defined above ---
                try:
                    popt, fit_line, angle_line, fit_label, r2, mse = fit_and_plot_tangent(all_angles, all_heights, X_test_angle, X_test_height)
                    if popt is not None:
                        fitted_params.append({'sd_mm': sd, 'd_t_mm': td, 'params': popt})
                        # Evaluate tangent fit for each (X_test_angle, Y_test_height) pair if available
                        if len(X_test_angle) > 0 and len(Y_test_height) > 0:
                            if 'shared_variables.results_table' in globals():
                                shared_variables.results_table.append({
                                    'Fold': None,
                                    'Model': f'Tangent (sd={sd}, d_t={td})',
                                    'Target': 'h_CAD_mm',
                                    'R2': r2,
                                    'MSE': mse
                                })



                        # # Plot tangent fit for this (sd, td)
                        # plt.figure(figsize=(8, 6))
                        # plt.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)
                        # plt.scatter(sdtd_all_angles, sdtd_all_heights, color='blue', alpha=0.5, label='Train Data Points')
                        # plt.xlabel('Measured Angle [°]', fontsize=14)
                        # plt.ylabel('Measured Height', fontsize=14)
                        # plt.title(f'Tangent Fit: Height vs Angle (sd_mm={sd}, d_t_mm={td})')
                        # plt.legend()
                        # plt.grid(True, linestyle='--', alpha=0.7)
                        # plt.tight_layout()
                        # plt.show()
                except Exception as e:
                    print(f"Tangent fit failed for sd_mm={sd}, d_t_mm={td}: {e}")

                # --- Use linear regression for height ---
                try:
                    if len(all_heights) > 1 and len(CAD_all_heights) > 1:
                        model_angle, model_height = plot_linear_regression_all(
                            all_angles, CAD_all_angles, all_angles, CAD_all_angles,
                            all_heights, CAD_all_heights, all_heights, CAD_all_heights,
                            X_test_angle, Y_test_angle, X_test_height, Y_test_height
                        )
                        # Store the linear regression parameters for height
                        slope_h = model_height.coef_[0]
                        intercept_h = model_height.intercept_
                        linear_params_h.append({'sd_mm': sd, 'd_t_mm': td, 'slope': slope_h, 'intercept': intercept_h})

                        # Store the linear regression parameters for angle
                        slope_a = model_angle.coef_[0]
                        intercept_a = model_angle.intercept_
                        linear_params_angle.append({'sd_mm': sd, 'd_t_mm': td, 'slope': slope_a, 'intercept': intercept_a})

                                                # Evaluate tangent fit for each (X_test_angle, Y_test_height) pair if available

                        # Store results for linear regression (height)
                        if 'shared_variables.results_table' in globals():
                            shared_variables.results_table.append({
                                'Fold': None,
                                'Model': f'Linear (Height, sd={sd}, d_t={td})',
                                'Target': 'h_CAD_mm',
                                'R2': r2_score(Y_test_height, model_height.predict(X_test_height.reshape(-1, 1))) if len(X_test_height) > 0 else None,
                                'MSE': mean_squared_error(Y_test_height, model_height.predict(X_test_height.reshape(-1, 1))) if len(X_test_height) > 0 else None
                            })
                            shared_variables.results_table.append({
                                'Fold': None,
                                'Model': f'Linear (Angle, sd={sd}, d_t={td})',
                                'Target': 'psi_CAD_deg',
                                'R2': r2_score(Y_test_angle, model_angle.predict(X_test_angle.reshape(-1, 1))) if len(X_test_angle) > 0 else None,
                                'MSE': mean_squared_error(Y_test_angle, model_angle.predict(X_test_angle.reshape(-1, 1))) if len(X_test_angle) > 0 else None
                            })

                except Exception as e:
                    print(f"Linear regression (height/angle) failed for sd_mm={sd}, d_t_mm={td}: {e}")

                # --- Group scatter and subplots ---
                plot_group_scatter_and_subplots(
                    groups=[g40, g70, g50, g60],
                    colors=['blue', 'green', 'red', 'orange'],
                    labels=['Group 40', 'Group 70', 'Group 50', 'Group 60'],
                    xlims=[(38.5, 39.5), (69.5, 70.5), (49.5, 50.5), (59.5, 60.5)],
                    ylims=[(24, 25), (82, 83), (35, 36), (51, 52)],
                    title_main=f'Scatter Plot for All Groups (sd_mm={sd}, d_t_mm={td})',
                    title_subplots='Subplot for'
                )
    return fitted_params, linear_params_h, linear_params_angle


def predict_and_plot_models(new_data, fitted_params, linear_params_h, linear_params_angle, prediction_table, fold_idx):
    """
    Predict using fitted tangent and linear models for new_data, and plot all linear models.
    """
    sd_val = float(new_data['sd_mm'].iloc[0])
    dt_val = float(new_data['d_t_mm'].iloc[0])

    # Predict with tangent model
    matched_fit = next((fit for fit in fitted_params if fit['sd_mm'] == sd_val and fit['d_t_mm'] == dt_val), None)
    if matched_fit:
        popt = matched_fit['params']
        pred_height_tan = tangent_func(new_data['psi_D_deg'].iloc[0], *popt)
        print(f"Tangent model (sd_mm={sd_val}, d_t_mm={dt_val}): {popt}, Predicted height: {pred_height_tan:.2f}")
        # Store the predicted height from the tangent model for new_data in the prediction_table
        prediction_table = store_predictions_table(
            prediction_table,
            fold_idx,
            f'Tangent model (sd_mm={sd_val}, d_t_mm={dt_val})',
            ['h_CAD_mm'],
            new_data,  # True value is not available, so we can store NaN or the input itself
            np.array([[pred_height_tan]]),
            new_data.index
        )

    else:
        print(f"No tangent model for sd_mm={sd_val}, d_t_mm={dt_val}")

    # Predict with linear regression (height)
    matched_linear_h = next((fit for fit in linear_params_h if fit['sd_mm'] == sd_val and fit['d_t_mm'] == dt_val), None)
    if matched_linear_h:
        pred_height_lin = matched_linear_h['slope'] * new_data['h_D_mm'].iloc[0] + matched_linear_h['intercept']
        print(f"Linear height model (sd_mm={sd_val}, d_t_mm={dt_val}): slope={matched_linear_h['slope']}, intercept={matched_linear_h['intercept']}, Predicted: {pred_height_lin:.2f}")

        prediction_table = store_predictions_table(
            prediction_table,
            fold_idx,
            f'Linear height model (sd_mm={sd_val}, d_t_mm={dt_val})',
            ['h_CAD_mm'],
            new_data,  # True value is not available, so we can store NaN or the input itself
            np.array([[pred_height_lin]]),
            new_data.index
        )
    else:
        print(f"No linear height model for sd_mm={sd_val}, d_t_mm={dt_val}")

    # Predict with linear regression (angle)
    matched_linear_angle = next((fit for fit in linear_params_angle if fit['sd_mm'] == sd_val and fit['d_t_mm'] == dt_val), None)
    if matched_linear_angle:
        pred_angle_lin = matched_linear_angle['slope'] * new_data['psi_D_deg'].iloc[0] + matched_linear_angle['intercept']
        print(f"Linear angle model (sd_mm={sd_val}, d_t_mm={dt_val}): slope={matched_linear_angle['slope']}, intercept={matched_linear_angle['intercept']}, Predicted: {pred_angle_lin:.2f}")
        prediction_table = store_predictions_table(
            prediction_table,
            fold_idx,
            f'Linear angle model (sd_mm={sd_val}, d_t_mm={dt_val})',
            ['psi_CAD_deg'],
            new_data,  # True value is not available, so we can store NaN or the input itself
            np.array([[pred_angle_lin]]),
            new_data.index
        )
    
    else:
        print(f"No linear angle model for sd_mm={sd_val}, d_t_mm={dt_val}")

    # # Plot all linear height models
    # if linear_params_h:
    #     plt.figure(figsize=(10, 6))
    #     for fit in linear_params_h:
    #         h = np.linspace(20, 90, 200)
    #         plt.plot(h, fit['slope'] * h + fit['intercept'], label=f"sd={fit['sd_mm']}, d_t={fit['d_t_mm']}")
    #     plt.xlabel('Height [mm]')
    #     plt.ylabel('Input Height (CAD) [mm]')
    #     plt.title('Linear Regression: CAD Height vs Measured Height')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.show()

    # # Plot all linear angle models
    # if linear_params_angle:
    #     plt.figure(figsize=(10, 6))
    #     for fit in linear_params_angle:
    #         a = np.linspace(20, 90, 200)
    #         plt.plot(a, fit['slope'] * a + fit['intercept'], label=f"sd={fit['sd_mm']}, d_t={fit['d_t_mm']}")
    #     plt.xlabel('Angle [º]')
    #     plt.ylabel('Input Angle (CAD) [º]')
    #     plt.title('Linear Regression: CAD Angle vs Measured Angle')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.7)
    #     plt.tight_layout()
    #     plt.show()
    return prediction_table


def plot_nn_hyperparameter_search(X_train, Y_train, fold_idx):
    param_grid = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
    'activation': ['relu'],
    'solver': ['adam', 'lbfgs'],
    'max_iter': [500],
    'random_state': [42]
    }

    mlp = MLPRegressor()
    grid = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, refit=True)
    grid.fit(X_train, Y_train)

    results = grid.cv_results_

    plt.figure(figsize=(12, 7))
    markers = ['o', 's', '^', 'D', 'x', '*', 'P', 'v']
    color_map = {
    ('adam', 500): 'blue',
    ('adam', 1000): 'green',
    ('adam', 2000): 'red',
    ('adam', 4000): 'purple',
    ('lbfgs', 500): 'orange',
    ('lbfgs', 1000): 'cyan',
    ('lbfgs', 2000): 'magenta',
    ('lbfgs', 4000): 'brown'
    }
    legend_labels = set()
    for idx, params in enumerate(results['params']):
        solver = params['solver']
        max_iter = params['max_iter']
        hls = str(params['hidden_layer_sizes'])
        score = results['mean_test_score'][idx]
        color = color_map.get((solver, max_iter), 'black')
        marker = markers[list(param_grid['max_iter']).index(max_iter) % len(markers)]
        label = f"{solver}, max_iter={max_iter}"
        if label not in legend_labels:
            plt.scatter(hls, score, color=color, marker=marker, label=label)
            legend_labels.add(label)
        else:
            plt.scatter(hls, score, color=color, marker=marker)
    plt.xlabel('hidden_layer_sizes')
    plt.ylabel('Mean CV R^2 Score')
    plt.title('MLPRegressor: Hyperparameter Comparison')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    mlp = grid.best_estimator_
    print(f"Best params: {mlp.get_params()}")
    print(f"Best CV R^2: {grid.best_score_:.3f}")

    # Store best estimator parameters in a table
    global best_params_table
    # Append best estimator parameters to global best_params_table
    best_params_row = {
        'hidden_layer_sizes': mlp.hidden_layer_sizes,
        'activation': mlp.activation,
        'solver': mlp.solver,
        'max_iter': mlp.max_iter,
        'best_cv_r2': grid.best_score_
    }
    if 'best_params_table' in globals() and isinstance(globals()['best_params_table'], pd.DataFrame):
        globals()['best_params_table'] = pd.concat([globals()['best_params_table'], pd.DataFrame([best_params_row])], ignore_index=True)
    else:
        globals()['best_params_table'] = pd.DataFrame([best_params_row])


    # Create a table of best parameters, ordered by best_cv_r2
    # Order all estimators in the grid from best to worst and store the same columns as best_params_table
    all_params_table = pd.DataFrame([
        {
            'fold': fold_idx,  # Use the fold index passed to the function
            'hidden_layer_sizes': params['hidden_layer_sizes'],
            'activation': params['activation'],
            'solver': params['solver'],
            'max_iter': params['max_iter'],
            'best_cv_r2': score
        }
        for params, score in zip(results['params'], results['mean_test_score'])
    ])

    all_params_table = all_params_table.sort_values(by='best_cv_r2', ascending=False).reset_index(drop=True)
    if 'all_params_table' in globals() and isinstance(globals()['all_params_table'], pd.DataFrame):
        globals()['all_params_table'] = pd.concat([globals()['all_params_table'], all_params_table], ignore_index=True)
    else:
        globals()['all_params_table'] = all_params_table


    return mlp, grid, best_params_table, all_params_table


def evaluate_and_plot_nn(mlp, X_train, Y_train, X_test, Y_test, output_targets, latex_labels, new_data, prediction_table, fold_idx):
    """
    Evaluate the best neural network model from GridSearchCV, print metrics, and plot predictions.
    """
    # Neural network model

    mlp.fit(X_train, Y_train)

    # Predict
    Y_pred_nn = mlp.predict(X_test)
    # Evaluation
    r2_nn_scores = []
    mse_nn_scores = []
    for i, target in enumerate(output_targets):
        r2_val = r2_score(Y_test.iloc[:, i], Y_pred_nn[:, i])
        mse_val = mean_squared_error(Y_test.iloc[:, i], Y_pred_nn[:, i])
        r2_nn_scores.append(r2_val)
        mse_nn_scores.append(mse_val)
        print(f"\nNeural Network - {target}")
        print("R^2 score:", r2_val)
        print("MSE:", mse_val)

        # Add results to model performance table
    if 'shared_variables.results_table' in globals():
        for i, target in enumerate(output_targets):
            shared_variables.results_table.append({
                'Fold': None,  # Fill with fold number if available in your context
                'Model': 'Neural Network',
                'Target': target,
                'R2': r2_nn_scores[i],
                'MSE': mse_nn_scores[i]
            })

    # Example prediction

    predicted_nn = mlp.predict(new_data)
    print(f"NN Predicted input angle: {predicted_nn[0][0]:.2f}")
    print(f"NN Predicted input height: {predicted_nn[0][1]:.2f}")

    # Only store the predicted values of the best estimator for psi_CAD_deg and h_CAD_mm
    prediction_table = store_predictions_table(
        prediction_table,
        fold_idx,
        'Neural Network',
        output_targets,
        pd.DataFrame(np.nan, index=new_data.index, columns=output_targets),  # True values unknown for new_data
        predicted_nn,
        new_data.index
    )

    # Plot NN predicted data against the rest of the test data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['psi_D_deg'], Y_test['psi_CAD_deg'], color='blue', label='Actual psi_CAD_deg')
    plt.scatter(X_test['psi_D_deg'], Y_pred_nn[:, 0], color='red', label='NN Predicted psi_CAD_deg', alpha=0.7)
    plt.scatter(new_data['psi_D_deg'], predicted_nn[0][0], color='green', label='NN Prediction (new data)', marker='x', s=100)
    plt.xlabel(latex_labels['psi_D_deg'], fontsize=14)
    plt.ylabel(latex_labels['psi_CAD_deg'], fontsize=14)
    plt.title('NN Predicted vs Actual Input Angle (psi_CAD_deg)', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test['h_D_mm'], Y_test['h_CAD_mm'], color='blue', label='Actual h_CAD_mm')
    plt.scatter(X_test['h_D_mm'], Y_pred_nn[:, 1], color='red', label='NN Predicted h_CAD_mm', alpha=0.7)
    plt.scatter(new_data['h_D_mm'], predicted_nn[0][1], color='green', label='NN Prediction (new data)', marker='x', s=100)
    plt.xlabel(latex_labels['h_D_mm'], fontsize=14)
    plt.ylabel(latex_labels['h_CAD_mm'], fontsize=14)
    plt.title('NN Predicted vs Actual Input Height (h_CAD_mm)', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # Example usage of store_predictions_table (add this where you want to store predictions for a model/fold)
     
    return prediction_table

def plot_group_scatter_and_subplots(groups, colors, labels, xlims, ylims, title_main, title_subplots):
    """
    Plot scatter and 4-subplot visualization for given group datasets.
    groups: list of DataFrames (g40, g70, g50, g60)
    colors: list of color strings
    labels: list of group labels
    xlims, ylims: list of axis limits for each subplot
    title_main: title for main scatter plot
    title_subplots: title prefix for subplots
    """


    # Main scatter plot
    # plt.figure(figsize=(10, 6))
    for group, color, label in zip(groups, colors, labels):
        plt.scatter(group['psi_D_deg'], group['h_D_mm'], color=color, label=label)
    plt.xlabel('Average Angle [°]', fontsize=14)
    plt.ylabel('Average Height', fontsize=14)
    plt.title(title_main)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Fit tangent function to all data
    all_angles = np.concatenate([g['psi_D_deg'].values for g in groups])
    all_heights = np.concatenate([g['h_D_mm'].values for g in groups])
    p0 = [30, 1, 0, 0]
    try:
        popt, _ = opt.curve_fit(tangent_func, all_angles, all_heights, p0=p0, maxfev=10000)
        fit_label = f'Best Fit: {popt[0]:.2f}·tan({popt[1]:.2f}·Angle + {popt[2]:.2f}) + {popt[3]:.2f}'
        angle_line = np.linspace(35, 75, 100)
        fit_line = tangent_func(angle_line, *popt)
        plt.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)
    except Exception as e:
        print(f"Tangent fit failed: {e}")

    # Reference line
    angle_line = np.linspace(35, 75, 100)
    height_line = 30 * np.tan(np.radians(angle_line))
    plt.plot(angle_line, height_line, color='black', linestyle='--', label='Height = 30·tan(Angle)')

    plt.xlabel('Average Angle [°]', fontsize=14)
    plt.ylabel('Average Height', fontsize=14)
    plt.title(title_main + ' (Switched Axes)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(20, 90)
    plt.xlim(35, 75)
    plt.tight_layout()
    plt.show()

    # 4-subplot visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (ax, group, color, label) in enumerate(zip(axes.flat, groups, colors, labels)):
        ax.scatter(group['psi_D_deg'], group['h_D_mm'], color=color, label=label)
        if 'fit_line' in locals():
            ax.plot(angle_line, fit_line, color='magenta', linestyle='-', label=fit_label)
        ax.plot(angle_line, height_line, color='black', linestyle='--', label='Height = 30·tan(Angle)')
        ax.set_xlabel('Average Angle [°]', fontsize=14)
        ax.set_ylabel('Average Height', fontsize=14)
        ax.set_title(f'{title_subplots} {label}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        if xlims[idx] is not None:
            ax.set_xlim(xlims[idx][0], xlims[idx][1])
        if ylims[idx] is not None:
            ax.set_ylim(ylims[idx][0], ylims[idx][1])
    plt.tight_layout()
    plt.show()

