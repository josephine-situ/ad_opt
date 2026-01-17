import pandas as pd
import argparse
from pathlib import Path
import numpy as np
import sys

def get_color_map(origins):
    # Pastel colors suitable for reading text
    palette = [
        '#FFB6C1', # LightPink
        '#98FB98', # PaleGreen
        '#87CEEB', # SkyBlue
        '#FFE4B5', # Moccasin
        '#DDA0DD', # Plum
        '#F0E68C', # Khaki
        '#E0FFFF', # LightCyan
        '#FFDAB9', # PeachPuff
        '#E6E6FA', # Lavender
        '#FF6961', # Pastel Red
    ]
    unique_origins = sorted(list(set(origins)))
    return {origin: palette[i % len(palette)] for i, origin in enumerate(unique_origins)}

def process_directory(input_dir_str):
    input_dir = Path(input_dir_str).resolve()
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    # Check for 'bids' subdirectory
    bids_dir = input_dir / "bids"
    if not bids_dir.exists():
        # Fallback: check if the input dir itself contains the files
        if list(input_dir.glob("optimized_costs_*.csv")):
            bids_dir = input_dir
        else:
            print(f"Error: No 'bids' subdirectory or optimized_costs csvs found in {input_dir}")
            return

    # Create output directory
    output_dir = input_dir / "formatted_bids_excel"
    output_dir.mkdir(exist_ok=True)
    print(f"Processing files from {bids_dir}")
    print(f"Outputting to {output_dir}")

    files = list(bids_dir.glob("optimized_costs_*.csv"))
    if not files:
        print("No optimized_costs_*.csv files found.")
        return

    print(f"Found {len(files)} files to process.")

    for file_path in files:
        try:
            df = pd.read_csv(file_path)
            
            # Check required columns
            required = ['Keyword', 'Origin', 'Optimal Cost', 't_Clicks_OptCost']
            missing = [c for c in required if c not in df.columns]
            if missing:
                print(f"  Skipping {file_path.name}: Missing columns {missing}")
                continue

            # Calculate CPC
            def calc_cpc(row):
                cost = row['Optimal Cost']
                clicks = row['t_Clicks_OptCost']
                if pd.isna(cost) or pd.isna(clicks):
                    return np.nan
                if clicks > 0:
                    return cost / clicks
                if cost > 0:
                    return np.inf
                return 0.0

            df['CPC'] = df.apply(calc_cpc, axis=1)
            
            # Sort by CPC ascending
            df = df.sort_values('CPC', ascending=True)

            # Prepare for Excel
            origins = df['Origin'].astype(str).fillna("Unknown").values
            df['Origin'] = origins
            unique_origins = sorted(list(set(origins)))
            color_map = get_color_map(unique_origins)

            # Remove unnecessary columns
            cols_to_remove = ["Actual Model Pred", "Diff"]
            df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])

            output_file = output_dir / file_path.with_suffix('.xlsx').name
            
            # Use xlsxwriter for formatting
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Bids')
                
                workbook = writer.book
                worksheet = writer.sheets['Bids']
                
                # Define formats
                header_fmt = workbook.add_format({'bold': True, 'bottom': 1, 'bg_color': '#DDDDDD', 'text_wrap': True})
                num_fmt = workbook.add_format({'num_format': '#,##0.00'})
                currency_fmt = workbook.add_format({'num_format': '$#,##0.00'})
                
                # Keyword formats (Origins)
                origin_fmts = {
                    origin: workbook.add_format({'bg_color': color}) 
                    for origin, color in color_map.items()
                }
                
                # Apply header format + column widths
                for col_num, value in enumerate(df.columns.values):
                    worksheet.write(0, col_num, value, header_fmt)
                    
                    # Set approximate column widths
                    if value == "Keyword":
                        worksheet.set_column(col_num, col_num, 40)
                    elif "Cost" in value or "CPC" in value or "Clicks" in value:
                        worksheet.set_column(col_num, col_num, 12)
                    elif value == "Origin":
                         worksheet.set_column(col_num, col_num, 15)
                    else:
                        worksheet.set_column(col_num, col_num, 15)

                # Set number formats for specific columns
                col_indices = {name: i for i, name in enumerate(df.columns)}
                

                worksheet.set_column(col_indices['Optimal Cost'], col_indices['Optimal Cost'], 12, currency_fmt)
                worksheet.set_column(col_indices['CPC'], col_indices['CPC'], 12, currency_fmt)
                worksheet.set_column(col_indices['t_Clicks_OptCost'], col_indices['t_Clicks_OptCost'], 12, num_fmt)
                worksheet.set_column(col_indices['Gurobi Pred'], col_indices['Gurobi Pred'], 12, num_fmt)
                worksheet.set_column(col_indices['Gurobi Pred over Base'], col_indices['Gurobi Pred over Base'], 12, num_fmt)

                # Apply color coding to Keyword column
                kw_col_idx = col_indices.get("Keyword", 0)
                
                # Iterate rows to write keywords with color
                for row_idx, (idx, row) in enumerate(df.iterrows()):
                    # Excel row = row_idx + 1 (0 is header)
                    excel_row = row_idx + 1
                    origin = row['Origin']
                    fmt = origin_fmts.get(origin)
                    
                    # Overwrite the Keyword cell with format
                    worksheet.write(excel_row, kw_col_idx, row['Keyword'], fmt)

                # Add Legend
                legend_start_col = len(df.columns) + 1
                worksheet.write(0, legend_start_col, "Legend: Origin", header_fmt)
                for i, origin in enumerate(unique_origins):
                    worksheet.write(i + 1, legend_start_col, origin, origin_fmts[origin])
                
                # worksheet.freeze_panes(1, 0)
                
            print(f"  Generated {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create formatted Excel files for bid optimization results.")
    parser.add_argument("input_dir", help="Path to the directory containing bid results")
    args = parser.parse_args()
    
    process_directory(args.input_dir)
