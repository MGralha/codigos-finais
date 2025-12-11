import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def convert_igs_to_text_and_split(file_path, split_word):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        # Split the content by the given word
        split_content = content.split(split_word)

        # Add the split word back to the beginning of each section (except the first)
        split_content = [split_content[0]] + [split_word + section for section in split_content[1:]]

        return split_content

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def remove_outliers(dataframe, col_x, col_y, q1, q2):
    """
    Removes outliers from the given DataFrame based on the IQR method for the specified columns.
    """
    Q1_x = dataframe[col_x].quantile(q1)
    Q3_x = dataframe[col_x].quantile(q2)
    IQR_x = Q3_x - Q1_x

    Q1_y = dataframe[col_y].quantile(q1)
    Q3_y = dataframe[col_y].quantile(q2)
    IQR_y = Q3_y - Q1_y

    lower_bound_x = Q1_x - 1.5 * IQR_x
    upper_bound_x = Q3_x + 1.5 * IQR_x

    lower_bound_y = Q1_y - 1.5 * IQR_y
    upper_bound_y = Q3_y + 1.5 * IQR_y

    filtered_data = dataframe[
        (dataframe[col_x] >= lower_bound_x) & (dataframe[col_x] <= upper_bound_x) &
        (dataframe[col_y] >= lower_bound_y) & (dataframe[col_y] <= upper_bound_y)
    ].reset_index(drop=True)

    return filtered_data


def calculate_threshold_index(dataset_table_part, thresholda, threshold=None, use_value=None):
    differencesx = dataset_table_part[1].diff().abs().dropna()
    differencesy = dataset_table_part[2].diff().abs().dropna()

    differences = (differencesy / differencesx).dropna()
    differencesangle = differences.diff().abs().dropna()
    
    threshold_indexd = None
    
    if use_value == 1:      
        
        # Calculate a moving average of differencesy
        window_size = 10  # Define the window size for the moving average
        moving_average = differencesy.rolling(window=window_size).mean()
        # Plot the moving average in order of the index
        # plt.figure(figsize=(10, 6))
        # plt.plot(moving_average, label='Moving Average', color='blue')
        # plt.axhline(y=0.035, color='red', linestyle='--', label='Threshold (0.035)')
        # plt.xlabel('Index')
        # plt.ylabel('Moving Average of DifferencesY')
        # plt.title('Moving Average of DifferencesY vs Index')
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # Find the index where the moving average is lower than 0.05
        threshold_indexd = moving_average[moving_average < 0.05].first_valid_index()
        if threshold_indexd is None:
            threshold_indexd = dataset_table_part[2].idxmax()
        elif threshold_indexd > 5 :
            threshold_indexd = threshold_indexd -5
        
    elif use_value == 2:
        # Find the index of the value with the highest value in the second column
        threshold_indexd = dataset_table_part[2].idxmax()
    else:
        threshold_indexd = differences[(differences > thresholda) & (differences.shift(-1) > thresholda)].first_valid_index()
        if threshold_indexd is None:
            threshold_indexd = threshold_indexd
        elif threshold_indexd > 5:
            threshold_indexd = threshold_indexd -5
        
        
    if threshold is not None and use_value is None:
        threshold_indexy = differencesy[(differencesy > threshold)].first_valid_index()
    elif threshold is not None and use_value == 1:
        threshold_indexy = differencesx[(differencesx > threshold)].first_valid_index()
    else:
        threshold_indexy = None



    if threshold_indexd is not None and threshold_indexy is not None:
        return min(threshold_indexd, threshold_indexy)
    elif threshold_indexd is not None:
        return threshold_indexd
    elif threshold_indexy is not None:
        return threshold_indexy
    else:
        return None
    
    
def process_threshold_index(dataset_table_part, threshold_index,name, peca, scan, iteracao, part, use_value=None):
    if threshold_index is not None:
        if use_value is None or use_value == 1:
            if use_value == 1: #nos casos dos angulos
                if threshold_index > 10:
                    filtered_rows = dataset_table_part.iloc[10:threshold_index]
                else:
                    filtered_rows = dataset_table_part.iloc[:threshold_index]
            else:  #nos casos dos horizontais
                filtered_rows = dataset_table_part.iloc[:threshold_index]
        elif use_value == 2: #nos casos dos fundos
            filtered_rows = dataset_table_part.iloc[threshold_index:]

        dataset_table_part = dataset_table_part.iloc[filtered_rows.index[-1] + 1:].reset_index(drop=True)
        filtered_rows = remove_outliers(filtered_rows, 1, 2, 0.325, 0.675)
        globals()[f"{name}_table_{peca}_{scan}_{iteracao}_{part}"] = filtered_rows
    return dataset_table_part

def divisaodedados(folder_path, output_file):
# Example usage

    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"Existing file {output_file} has been removed.")
    else:
        print(f"No existing file named {output_file} found.")
        
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.IGS')]

    for i, file_path in enumerate(file_paths):
        print(f"Processing file: {file_path}")
        
        file_name_parts = os.path.basename(file_path).split('_')    
        peca = file_name_parts[1]
        scan = file_name_parts[2]
        
        split_word = '2DSKETCH'
        sections = convert_igs_to_text_and_split(file_path, split_word)

        lines = sections[-1].splitlines()[4:]
        lines = [line for line in lines if not line.startswith('0.;')]

        last_110_index = max((i for i, line in enumerate(lines) if line.startswith('110')), default=-1)+1
        if last_110_index != -1:
            lines = lines[:last_110_index + 1]

        indices_124 = [0] + [i for i, line in enumerate(lines) if line.startswith('124,')]
        indices_402 = [i for i, line in enumerate(lines) if line.startswith('402,')]

        datasets = []
        iteracao=1
  
        
        for start_index in indices_124:

            end_index = next((i for i in indices_402 if i > start_index), len(lines))
            dataset_lines = lines[start_index+2:end_index]
            datasets.append(dataset_lines)

            dataset_table_name = f"dataset_table_{peca}_{scan}_{iteracao}"
            dataset_table = pd.DataFrame(dataset_lines)

            dataset_table[[0, 1, 2, 3]] = dataset_table[0].str.split(',', n=3, expand=True)
            dataset_table = dataset_table.drop(columns=[0, 3])
            
            # Order the points by ascending order of the first column (column 1) without losing the respective value of the second column
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            dataset_table[[1, 2]] = dataset_table[[1, 2]].apply(pd.to_numeric, errors='coerce')
            dataset_table = dataset_table.sort_values(by=1, ascending=True).reset_index(drop=True)
            
            # Check if dataset_table consists only of zeros in columns 1 and 2
            if (dataset_table[1].fillna(0) == 0).all() and (dataset_table[2].fillna(0) == 0).all():
                print(f"Dataset {dataset_table_name} consists only of zeros. Skipping to next start index.")
                continue                
                
            else:
                globals()[dataset_table_name] = dataset_table

                dataset_table = dataset_table[dataset_table[2] >= -0.1].reset_index(drop=True)
                nearest_to_zero_index = (dataset_table[1] - 0).abs().idxmin(axis=0, skipna=True)

                
                dataset_table_part1 = dataset_table.iloc[:nearest_to_zero_index + 1].reset_index(drop=True)
                dataset_table_part2 = dataset_table.iloc[nearest_to_zero_index + 1:].reset_index(drop=True)

                # Ensure column 1 has only positive values
                dataset_table_part1[1] = dataset_table_part1[1].abs()
                dataset_table_part2[1] = dataset_table_part2[1].abs()

                
                dataset_table_part1 = dataset_table_part1.sort_values(by=1, ascending=False).reset_index(drop=True)
                dataset_table_part2 = dataset_table_part2.sort_values(by=1, ascending=False).reset_index(drop=True)

                globals()[f"{dataset_table_name}_part1"] = dataset_table_part1
                globals()[f"{dataset_table_name}_part2"] = dataset_table_part2
                

                for part in ['part1', 'part2']:
                    dataset_table_name = f"dataset_table_{peca}_{scan}_{iteracao}_{part}"
                    dataset_table_part = globals()[dataset_table_name]

                    # Divide the dataset into three parts based on the first column's values
                    ranges = [(75, 52), (48, 22), (18, 0)]
                    colors = ['red', 'blue', 'green']  # Define colors for each range
                    #plt.figure(figsize=(10, 6))  # Create a new figure for the plot

                    medium_values = {}  # Dictionary to store medium values for ranges
                    slope = None  # Variable to store the slope for the 48 to 22 range

                    for (lower, upper), color in zip(ranges, colors):
                        range_filtered_rows = dataset_table_part[
                            (dataset_table_part[1] <= lower) & (dataset_table_part[1] > upper)
                        ].reset_index(drop=True)

                        # Remove outliers for the filtered rows
                        range_filtered_rows = remove_outliers(range_filtered_rows, 1, 2, 0.325, 0.675)

                        # Save the filtered rows to a global variable
                        globals()[f"{dataset_table_name}_{lower}_{upper}"] = range_filtered_rows

                        # Plot the filtered rows
                        # plt.scatter(range_filtered_rows[1], range_filtered_rows[2], label=f"{lower} > x > {upper}", color=color, alpha=1)

                        # Calculate medium value for the 75 to 52 and 18 to 0 ranges
                        if (lower, upper) in [(75, 52), (18, 0)]:
                            medium_values[f"{lower}_{upper}"] = range_filtered_rows[2].mean()

                        # Calculate the slope for the 48 to 22 range
                        if (lower, upper) == (48, 22) and not range_filtered_rows.empty:
                            x = range_filtered_rows[1]
                            y = range_filtered_rows[2]
                            if len(x) > 1:  # Ensure there are enough points to calculate the slope
                                slope = np.polyfit(x, y, 1)[0]  # Linear fit, slope is the first coefficient
                                slope = abs(np.degrees(np.arctan(slope)))  # Calculate the angle in the first quadrant
                    
                
                    # plt.xlabel('Column 1')
                    # plt.ylabel('Column 2')
                    # plt.title(f'Data Ranges for {dataset_table_name}')
                    # plt.legend()
                    # plt.grid(True)
                    # plt.show()

                    # Print the calculated medium values and slope
                    print(f"Medium values for ranges: {medium_values}")
                    if slope is not None:
                        print(f"Slope for the 48 to 22 range: {slope}")
                        
                        # Calculate the height by subtracting the first value of medium values from the second
                        if "75_52" in medium_values and "18_0" in medium_values:
                            height = medium_values["18_0"] - medium_values["75_52"]
                        else:
                            height = None
                        
                    # Create a dataset with the first column as the angle and the second as the height
                    angle_height_dataset = pd.DataFrame({
                        'Angle (degrees)': [slope],
                        'Height': [height],
                        'Peça': [peca],
                        'Scan': [scan],
                        'Iteração': [iteracao],
                        'Part': [part]
                    })

                
                    

                    # Append the data to the CSV file, creating it if it doesn't exist
                    if not os.path.exists(output_file):
                        angle_height_dataset.to_csv(output_file, index=False, mode='w')
                    else:
                        angle_height_dataset.to_csv(output_file, index=False, mode='a', header=False)

                    print(f"Angle-Height dataset for peça {peca} appended to {output_file}")
                    
                iteracao = iteracao  +1

def colors(custom_colors,i):
    return custom_colors[i % len(custom_colors)]