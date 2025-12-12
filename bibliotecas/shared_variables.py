import pandas as pd

results_table = []
prediction_table = []

peça_order = ['LRND', 'LRD', 'LRS', 'NN(8)', 'NN(16)']
new_data = pd.DataFrame([{           
        'psi_D_deg': 40,
        'h_D_mm': 25.17,
        'sd_mm': 0.7,
        'd_t_mm': 12,
        'd_D_mm': 20
    }])

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

latex_labels = {
    'psi_D_deg': r'$\Psi_m \, [^o]$',
    'h_D_mm': r'$h_m \, [mm]$',
    'diam_D_mm': r'$d_m \, [mm]$',
    'sd_mm': r'$sd \, [mm]$',
    'd_t_mm': r'$d_t \, [mm]$',
    'psi_CAD_deg': r'$\Psi_{CAD} \, [^o]$',
    'h_CAD_mm': r'$h_{CAD} \, [mm]$'
}

