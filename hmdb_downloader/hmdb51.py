import os
import rarfile
from multiprocessing import Pool, cpu_count

def extract_and_delete_rar(rar_path):
    try:
        rar_file = os.path.basename(rar_path)
        directory = os.path.dirname(rar_path)
        
        # Open and extract RAR file
        with rarfile.RarFile(rar_path) as rf:
            print(f"Extracting {rar_file}...")
            rf.extractall(directory)
            print(f"Extraction of {rar_file} completed.")
        
        # Delete the original RAR file
        os.remove(rar_path)
        print(f"Deleted {rar_file}")
        return True
    except rarfile.Error as e:
        print(f"Error processing {rar_file}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error with {rar_file}: {e}")
        return False

def process_directory(directory_path, max_processes=10):
    # Check if directory exists
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # Get all RAR files in directory
    rar_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                if f.endswith('.rar')]
    
    if not rar_files:
        print("No RAR files found in directory.")
        return
    
    # Determine number of processes to use
    num_processes = min(max_processes, len(rar_files))
    print(f"Processing {len(rar_files)} RAR files using {num_processes} processes")
    
    # Create pool and map function to all RAR files
    with Pool(processes=num_processes) as pool:
        results = pool.map(extract_and_delete_rar, rar_files)
    
    # Print summary
    successful = results.count(True)
    print(f"Completed: {successful}/{len(rar_files)} files successfully processed")

# Hardcoded directory path
directory_path = "/Users/shubhamrandive/Documents/codes/Actrix/data/hmdb51/hmdb51_org"

if __name__ == "__main__":
    # This conditional is important for multiprocessing on Windows
    process_directory(directory_path, max_processes=10)