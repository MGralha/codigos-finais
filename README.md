"Development of a Machine Learning framework for geometric compensation of Single Point Incremental Forming process"
Master's Thesis codes, by Margarida Ferreira Gralha

In the "codigos finais" folder, the user will find the following .py codes:
 - "compare compensated with non compensated" - compares the values of height and angle of the compensated and uncompensated parts with the objective geometry to be produced
 - "gauge_block_script" - defines the values of height and width of the scanned gauge block and compares it with the obtained measurements made with precision tools
 - "maincode" - Trains, validates and tests the models with the initial non compensated 32 parts
 - "maincode_updated" - Trains, validates and tests the models with the full dataset, including compensated parts (37 parts in total)
 - "red_convergence_analysis" - Convergence analysis to define the best reduction factor for the mesh
 - "res_convergence_analysis" - Convergence analysis to define the best resolution factor for the mesh

In all the codes, the user must insert in the base_path variable the complete folder path until arriving to "codigos finais". 
In the current state, the pathway is "C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais", and is presented has follows:
base_path = r'C:\Users\pucag\OneDrive - Universidade de Lisboa\Desktop\Dissertacao\codigos finais'


These codes either make use or output values from/to the following folders:
Inputs
 - "bibliotecas" - functions generated for the multiple codes
 - "csv_files" - input csv files are stored 
 - "sketch calco" - .igs files of the gauge block to convert into a dataset
 - "sketch compensados" - .igs files of the compensated parts to convert into a dataset
 - "sketch nao compencados" - .igs files of the non compensated parts to convert into a dataset
 - "sketch red" - .igs files of the same point cloud with different reduction factors 
 - "sketch res" - .igs files of the same point cloud with different facet resolutions
 - "time" - .txt files with the time to render the point clouds for the convergence analysis

Outputs
 - "results" - where the resulting CAD predictions and MSE from the models are stored
