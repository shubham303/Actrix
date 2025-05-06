import os
import subprocess
import concurrent.futures
from pathlib import Path

def convert_avi_to_mp4(avi_file):
    """Convert an AVI file to MP4 and delete the original AVI file."""
    mp4_file = avi_file.with_suffix('.mp4')
    
    # Skip if MP4 already exists
    if mp4_file.exists():
        print(f"Skipping {avi_file.name} - MP4 already exists")
        return False
    
    try:
        print(f"Converting {avi_file.name} to MP4...")
        
        # FFmpeg command to convert AVI to MP4
        cmd = [
            'ffmpeg',
            '-i', str(avi_file),
            '-c:v', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-y',
            str(mp4_file)
        ]
        
        # Run FFmpeg command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Delete the original AVI file if conversion was successful
        if mp4_file.exists():
            avi_file.unlink()
            print(f"Successfully converted {avi_file.name} and deleted original")
            return True
        else:
            print(f"MP4 file not created for {avi_file.name}")
            return False
    
    except subprocess.CalledProcessError as e:
        print(f"Error converting {avi_file.name}: {e}")
        print(f"FFmpeg error: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error with {avi_file.name}: {e}")
        return False

def find_and_convert_avi_files(root_dir, max_workers=4):
    """Recursively find all AVI files and convert them to MP4."""
    root_path = Path(root_dir)
    
    # Check if directory exists
    if not root_path.is_dir():
        print(f"Directory {root_dir} does not exist.")
        return
    
    # Find all AVI files recursively
    avi_files = list(root_path.glob('**/*.avi'))
    
    if not avi_files:
        print("No AVI files found.")
        return
    
    print(f"Found {len(avi_files)} AVI files")
    
    # Convert files using thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(convert_avi_to_mp4, avi_files))
    
    # Print summary
    successful = results.count(True)
    print(f"Conversion complete: {successful}/{len(avi_files)} files successfully converted and original files deleted")

if __name__ == "__main__":
    # Directory path to search for AVI files
    directory_path = "/Users/shubhamrandive/Documents/codes/Actrix/data/hmdb51/videos"
    
    # Run the conversion with 4 parallel workers
    find_and_convert_avi_files(directory_path, max_workers=4)