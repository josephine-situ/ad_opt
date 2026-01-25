import os
import shutil
import glob
from datetime import datetime, timedelta

def create_exp11():
    base_path = r'c:\Users\jsitu\OneDrive\Documents\Courses\Research\ad_opt\opt_results\backtests'
    exp10_path = os.path.join(base_path, 'exp10')
    exp11_path = os.path.join(base_path, 'exp11')

    if os.path.exists(exp11_path):
        print(f"Removing existing {exp11_path}")
        shutil.rmtree(exp11_path)
    
    os.makedirs(exp11_path)
    print(f"Created {exp11_path}")

    # Copy cache folder
    exp10_cache = os.path.join(exp10_path, 'cache')
    exp11_cache = os.path.join(exp11_path, 'cache')
    if os.path.exists(exp10_cache):
        print(f"Copying cache from {exp10_cache} to {exp11_cache}")
        shutil.copytree(exp10_cache, exp11_cache)
    else:
        print(f"No cache folder found at {exp10_cache}")

    # Process budget folders
    budget_folders = glob.glob(os.path.join(exp10_path, 'budget_*'))
    
    for budget_folder in budget_folders:
        folder_name = os.path.basename(budget_folder)
        print(f"Processing {folder_name}...")
        
        src_bids_dir = os.path.join(budget_folder, 'bids')
        dest_bids_dir = os.path.join(exp11_path, folder_name, 'bids')
        
        if not os.path.exists(src_bids_dir):
            print(f"Skipping {folder_name}, no bids directory.")
            continue
            
        os.makedirs(dest_bids_dir)
        
        # Get list of bid files and sort them
        bid_files = sorted(glob.glob(os.path.join(src_bids_dir, 'optimized_costs_*.csv')))
        
        # Parse dates
        file_date_map = {}
        for f in bid_files:
            # optimized_costs_YYYY-MM-DD.csv
            basename = os.path.basename(f)
            date_str = basename.replace('optimized_costs_', '').replace('.csv', '')
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                file_date_map[dt] = f
            except ValueError:
                print(f"Skipping file with bad date format: {basename}")
        
        dates = sorted(file_date_map.keys())
        if not dates:
            continue
            
        current_ref_date = None
        current_ref_file = None
        
        # Assume first date is start of a week
        # We need to reset the ref file every 7 days relative to the start, 
        # OR every Monday?
        # User example: 2025-12-01 (Monday) -> 2025-12-07 (Sunday) use 2025-12-01
        # This aligns with ISO weeks or just chunks of 7 starting from the first day.
        # Since 2025-12-01 is a Monday, both logic work same.
        
        start_date = dates[0]
        
        for dt in dates:
            days_diff = (dt - start_date).days
            week_idx = days_diff // 7
            
            # The start of the current chunk
            chunk_start_dt = start_date + timedelta(days=week_idx * 7)
            
            # If the calculated chunk start exists in our files, use it. 
            # If not (e.g. gaps), we might need to be careful. 
            # Assuming contiguous dates for now based on file listing.
            
            if chunk_start_dt in file_date_map:
                ref_file = file_date_map[chunk_start_dt]
            else:
                # Fallback, though we shouldn't hit this if contiguous
                ref_file = file_date_map[dt] # Just use own file if start of week missing?
                print(f"Warning: Start of week {chunk_start_dt} missing, using {dt} for itself.")

            dest_filename = f'optimized_costs_{dt.strftime("%Y-%m-%d")}.csv'
            dest_file_path = os.path.join(dest_bids_dir, dest_filename)
            
            # Copy content of ref_file to dest_file_path
            shutil.copy2(ref_file, dest_file_path)
            # print(f"Created {dest_filename} using content from {os.path.basename(ref_file)}")

    print("Experiment exp11 creation complete.")

if __name__ == "__main__":
    create_exp11()
