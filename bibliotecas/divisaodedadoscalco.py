import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import divisaodedados

def angle_between_slopes(m1, m2):
    # Returns the angle in degrees between two slopes
    if np.isnan(m1) or np.isnan(m2):
        return np.nan
    angle_rad = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))
    return np.degrees(angle_rad)

def process_igs_files_and_generate_csv(folder_path, namecsv):
    output_file = "angle_height_dataset_all_pecas.csv"
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} has been removed.")
    else:
        print(f"No existing file named {output_file} found.")
        
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.IGS')]

    for i, file_path in enumerate(file_paths):
        print(f"Processing file: {file_path}")
        
        peca = os.path.basename(file_path).replace('.', '_').split('_')    
        
        split_word = '2DSKETCH'
        sections = divisaodedados.convert_igs_to_text_and_split(file_path, split_word)

        lines = sections[-1].splitlines()[4:]
        lines = [line for line in lines if not line.startswith('0.;')]

        last_110_index = max((i for i, line in enumerate(lines) if line.startswith('110')), default=-1)+1
        if last_110_index != -1:
            lines = lines[:last_110_index + 1]

        indices_124 = [0] + [i for i, line in enumerate(lines) if line.startswith('124,')]
        indices_402 = [i for i, line in enumerate(lines) if line.startswith('402,')]

        datasets = []
        iteracao=1

        angle_height_dataset = []

        for start_index in indices_124[:9]:
            end_index = next((i for i in indices_402 if i > start_index), len(lines))
            dataset_lines = lines[start_index+2:end_index]
            datasets.append(dataset_lines)

            dataset_table_name = f"dataset_table_{peca}_{iteracao}"
            dataset_table = pd.DataFrame(dataset_lines)

            dataset_table[[0, 1, 2, 3]] = dataset_table[0].str.split(',', n=3, expand=True)
            dataset_table = dataset_table.drop(columns=[0, 3])
            
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            dataset_table[[1, 2]] = dataset_table[[1, 2]].apply(pd.to_numeric, errors='coerce')
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            if (dataset_table[1].fillna(0) == 0).all() and (dataset_table[2].fillna(0) == 0).all():
                print(f"Dataset {dataset_table_name} consists only of zeros. Skipping to next start index.")
                continue                
            else:
                globals()[dataset_table_name] = dataset_table
                
                min_col1 = dataset_table[1].min()
                dataset_table[1] = dataset_table[1] - min_col1

                dataset_table_name = f"dataset_table_{peca}_{iteracao}"
                dataset_table_part = globals()[dataset_table_name]

                ranges = [(0, 7), (73, 80), (13, 67), (3,20)]
                colors = ['red', 'blue', 'green']

                interval1 = dataset_table_part[(dataset_table_part[1] >= ranges[0][0]) & (dataset_table_part[1] <= ranges[0][1])]
                interval2 = dataset_table_part[(dataset_table_part[1] >= ranges[1][0]) & (dataset_table_part[1] <= ranges[1][1])]
                intervals_data = pd.concat([interval1, interval2])

                X = intervals_data[1].values.reshape(-1, 1)
                y = intervals_data[2].values
                reg = LinearRegression().fit(X, y)
                slope_lr = reg.coef_[0]
                intercept_lr = reg.intercept_

                theta = -np.arctan(slope_lr)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                points = dataset_table_part[[1, 2]].values
                rotated_points = (points - [0, intercept_lr]) @ R.T
                dataset_table_part[1] = rotated_points[:, 0]
                dataset_table_part[2] = rotated_points[:, 1]

                third_interval = dataset_table_part[
                    (dataset_table_part[1] >= ranges[2][0]) & (dataset_table_part[1] <= ranges[2][1])
                ].reset_index(drop=True)
                third_interval_filtered = divisaodedados.remove_outliers(third_interval, 1, 2, 0.325, 0.675)
                height = third_interval_filtered[2].mean()
                print(f"height ({ranges[2][0]}, {ranges[2][1]}): {height}")

                medium_values = {}

                fourth_interval = dataset_table_part[
                    (dataset_table_part[2] >= ranges[3][0]) & (dataset_table_part[2] <= ranges[3][1])
                ].reset_index(drop=True)
                fourth_interval_filtered = divisaodedados.remove_outliers(fourth_interval, 1, 2, 0.325, 0.675)
                midpoint = (ranges[2][0] + ranges[2][1]) / 2
                subinterval1 = fourth_interval_filtered[fourth_interval_filtered[1] < midpoint]
                subinterval2 = fourth_interval_filtered[fourth_interval_filtered[1] >= midpoint]

                avg_col1_subinterval1 = subinterval1[1].mean() if not subinterval1.empty else np.nan
                avg_col1_subinterval2 = subinterval2[1].mean() if not subinterval2.empty else np.nan

                print(f"Average column 1 value in 4th range, subinterval 1 ({ranges[3][0]} to {midpoint}): {avg_col1_subinterval1}")
                print(f"Average column 1 value in 4th range, subinterval 2 ({midpoint} to {ranges[3][1]}): {avg_col1_subinterval2}")
                
                width = abs(avg_col1_subinterval2 - avg_col1_subinterval1)
                print(f"Width between subintervals: {width}")

                if not subinterval1.empty and len(subinterval1) > 1:
                    reg_sub1 = LinearRegression().fit(subinterval1[1].values.reshape(-1, 1), subinterval1[2].values)
                    slope_sub1 = reg_sub1.coef_[0]
                else:
                    slope_sub1 = np.nan

                if not third_interval_filtered.empty and len(third_interval_filtered) > 1:
                    reg_third = LinearRegression().fit(third_interval_filtered[1].values.reshape(-1, 1), third_interval_filtered[2].values)
                    slope_third = reg_third.coef_[0]
                else:
                    slope_third = np.nan

                if not subinterval2.empty and len(subinterval2) > 1:
                    reg_sub2 = LinearRegression().fit(subinterval2[1].values.reshape(-1, 1), subinterval2[2].values)
                    slope_sub2 = reg_sub2.coef_[0]
                else:
                    slope_sub2 = np.nan

                angle_lr_sub1 = angle_between_slopes(slope_lr, slope_sub1)
                angle_sub1_third = angle_between_slopes(slope_sub1, slope_third)
                angle_third_sub2 = angle_between_slopes(slope_third, slope_sub2)
                angle_sub2_lr = angle_between_slopes(slope_sub2, slope_lr)

                print(f"Angle between linear regression (ranges 1&2) and subinterval1: {angle_lr_sub1:.2f} degrees")
                print(f"Angle between subinterval1 and third interval: {angle_sub1_third:.2f} degrees")
                print(f"Angle between third interval and subinterval2: {angle_third_sub2:.2f} degrees")
                print(f"Angle between subinterval2 and linear regression (ranges 1&2): {angle_sub2_lr:.2f} degrees")

                plt.figure(figsize=(10, 6))
                plt.scatter(dataset_table_part[1], dataset_table_part[2], s=10, color='black', label='Rotated Points')
                plt.scatter(third_interval_filtered[1], third_interval_filtered[2], color='orange', label='3rd Interval Filtered', s=30)
                plt.scatter(fourth_interval_filtered[1], fourth_interval_filtered[2], color='purple', label='4th Interval Filtered', s=30)
                plt.xlabel('Column 1 (Rotated)')
                plt.ylabel('Column 2 (Rotated)')
                plt.title(f'Rotated Points for {dataset_table_name}')
                plt.legend()
                plt.grid(True)
                # plt.show()

                angle_height_dataset.append({
                    'Angle1': angle_lr_sub1,
                    'Angle2': angle_sub1_third,
                    'Angle3': angle_third_sub2,
                    'Angle4': angle_sub2_lr,
                    'Height': height,
                    'Width': width,
                    'Peça': peca,
                    'Iteração': iteracao,
                })

        for start_index in indices_124[9:]:
            end_index = next((i for i in indices_402 if i > start_index), len(lines))
            dataset_lines = lines[start_index+2:end_index]
            datasets.append(dataset_lines)

            dataset_table_name = f"dataset_table_{peca}_{iteracao}"
            dataset_table = pd.DataFrame(dataset_lines)

            dataset_table[[0, 1, 2, 3]] = dataset_table[0].str.split(',', n=3, expand=True)
            dataset_table = dataset_table.drop(columns=[0, 3])
            
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            dataset_table[[1, 2]] = dataset_table[[1, 2]].apply(pd.to_numeric, errors='coerce')
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            if (dataset_table[1].fillna(0) == 0).all() and (dataset_table[2].fillna(0) == 0).all():
                print(f"Dataset {dataset_table_name} consists only of zeros. Skipping to next start index.")
                continue                
            else:
                globals()[dataset_table_name] = dataset_table
                
                min_col1 = dataset_table[1].min()
                dataset_table[1] = dataset_table[1] - min_col1

                dataset_table_name = f"dataset_table_{peca}_{iteracao}"
                dataset_table = globals()[dataset_table_name]

                ranges = [(5, 15), (184, 192), (35, 165), (3,20)]
                colors = ['red', 'blue', 'green']

                third_interval = dataset_table[(dataset_table[1] >= ranges[2][0]) & (dataset_table[1] <= ranges[2][1])]
                third_interval_filtered = divisaodedados.remove_outliers(third_interval, 1, 2, 0.325, 0.675)

                X = third_interval_filtered[1].values.reshape(-1, 1)
                y = third_interval_filtered[2].values
                if len(X) > 1:
                    reg = LinearRegression().fit(X, y)
                    slope_lr = reg.coef_[0]
                    intercept_lr = reg.intercept_
                else:
                    slope_lr = 0
                    intercept_lr = 0

                theta = -np.arctan(slope_lr)
                R = np.array([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta),  np.cos(theta)]])
                points = dataset_table[[1, 2]].values
                rotated_points = (points - [0, intercept_lr]) @ R.T
                dataset_table[1] = rotated_points[:, 0]
                dataset_table[2] = rotated_points[:, 1]

                if len(dataset_table) > 20:
                    dataset_table = dataset_table.iloc[10:-10].reset_index(drop=True)

                first_point = dataset_table.iloc[0][[1, 2]].values
                dataset_table[1] = dataset_table[1] - first_point[0]
                dataset_table[2] = dataset_table[2] - first_point[1]

                third_interval = dataset_table[
                    (dataset_table[1] >= ranges[2][0]) & (dataset_table[1] <= ranges[2][1])
                ].reset_index(drop=True)
                third_interval_filtered = divisaodedados.remove_outliers(third_interval, 1, 2, 0.325, 0.675)
                height = third_interval_filtered[2].mean()
                print(f"Average value of the second column in the 3rd interval ({ranges[2][0]}, {ranges[2][1]}): {height}")

                medium_values = {}

                fourth_interval = dataset_table[
                    (dataset_table[2] >= ranges[3][0]) & (dataset_table[2] <= ranges[3][1])
                ].reset_index(drop=True)
                fourth_interval_filtered = divisaodedados.remove_outliers(fourth_interval, 1, 2, 0.325, 0.675)
                midpoint = (ranges[2][0] + ranges[2][1]) / 2
                subinterval1 = fourth_interval_filtered[fourth_interval_filtered[1] < midpoint]
                subinterval2 = fourth_interval_filtered[fourth_interval_filtered[1] >= midpoint]

                avg_col1_subinterval1 = subinterval1[1].mean() if not subinterval1.empty else np.nan
                avg_col1_subinterval2 = subinterval2[1].mean() if not subinterval2.empty else np.nan

                print(f"Average column 1 value in 4th range, subinterval 1 ({ranges[3][0]} to {midpoint}): {avg_col1_subinterval1}")
                print(f"Average column 1 value in 4th range, subinterval 2 ({midpoint} to {ranges[3][1]}): {avg_col1_subinterval2}")
                
                width = abs(avg_col1_subinterval2 - avg_col1_subinterval1)

                ranges = [(avg_col1_subinterval1-15, avg_col1_subinterval1-3), (avg_col1_subinterval2+3, avg_col1_subinterval2+15), (35, 165), (3,20)]

                interval1 = dataset_table[(dataset_table[1] >= ranges[0][0]) & (dataset_table[1] <= ranges[0][1])]
                interval2 = dataset_table[(dataset_table[1] >= ranges[1][0]) & (dataset_table[1] <= ranges[1][1])]
                intervals_data = pd.concat([interval1, interval2])

                X = intervals_data[1].values.reshape(-1, 1)
                y = intervals_data[2].values
                reg = LinearRegression().fit(X, y)
                slope_lr = reg.coef_[0]
                intercept_lr = reg.intercept_

                if not subinterval1.empty and len(subinterval1) > 1:
                    reg_sub1 = LinearRegression().fit(subinterval1[1].values.reshape(-1, 1), subinterval1[2].values)
                    slope_sub1 = reg_sub1.coef_[0]
                else:
                    slope_sub1 = np.nan

                if not third_interval_filtered.empty and len(third_interval_filtered) > 1:
                    reg_third = LinearRegression().fit(third_interval_filtered[1].values.reshape(-1, 1), third_interval_filtered[2].values)
                    slope_third = reg_third.coef_[0]
                else:
                    slope_third = np.nan

                if not subinterval2.empty and len(subinterval2) > 1:
                    reg_sub2 = LinearRegression().fit(subinterval2[1].values.reshape(-1, 1), subinterval2[2].values)
                    slope_sub2 = reg_sub2.coef_[0]
                else:
                    slope_sub2 = np.nan

                angle_lr_sub1 = angle_between_slopes(slope_lr, slope_sub1)
                angle_sub1_third = angle_between_slopes(slope_sub1, slope_third)
                angle_third_sub2 = angle_between_slopes(slope_third, slope_sub2)
                angle_sub2_lr = angle_between_slopes(slope_sub2, slope_lr)

                print(f"Angle between linear regression (ranges 1&2) and subinterval1: {angle_lr_sub1:.2f} degrees")
                print(f"Angle between subinterval1 and third interval: {angle_sub1_third:.2f} degrees")
                print(f"Angle between third interval and subinterval2: {angle_third_sub2:.2f} degrees")
                print(f"Angle between subinterval2 and linear regression (ranges 1&2): {angle_sub2_lr:.2f} degrees")

                plt.figure(figsize=(10, 6))
                plt.scatter(dataset_table[1], dataset_table[2], s=10, color='black', label='Rotated Points')
                plt.scatter(third_interval_filtered[1], third_interval_filtered[2], color='orange', label='3rd Interval Filtered', s=30)
                plt.scatter(fourth_interval_filtered[1], fourth_interval_filtered[2], color='purple', label='4th Interval Filtered', s=30)
                plt.xlabel('Column 1 (Rotated)')
                plt.ylabel('Column 2 (Rotated)')
                plt.title(f'Rotated Points for {dataset_table_name}')
                plt.legend()
                plt.grid(True)
                # plt.show()

                angle_height_dataset.append({
                    'Angle1': angle_lr_sub1,
                    'Angle2': angle_sub1_third,
                    'Angle3': angle_third_sub2,
                    'Angle4': angle_sub2_lr,
                    'Height': height,
                    'Width': width,
                    'Peça': peca,
                    'Iteração': iteracao,
                })
                
        output_file = namecsv
        angle_height_df = pd.DataFrame(angle_height_dataset)
        if not os.path.exists(output_file):
            angle_height_df.to_csv(output_file, index=False, mode='w')
        else:
            angle_height_df.to_csv(output_file, index=False, mode='a', header=False)

        print(f"Angle-Height dataset for peça {peca} appended to {output_file}")
            
        iteracao = iteracao  +1

