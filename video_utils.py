import cv2
import os

def convert_avi_to_mp4(video_path, output_path=None):
    """
    Convert AVI video to MP4 format if input is AVI.
    
    Args:
        video_path (str): Path to input video file
        output_path (str, optional): Path to save converted MP4 file. If None, saves in same directory.
    
    Returns:
        str: Path to the output video file (original path if no conversion needed)
    """
    # Check if input is AVI
    if not video_path.lower().endswith('.avi'):
        return video_path
        
    # Generate output path if not provided
    if output_path is None:
        output_path = os.path.splitext(video_path)[0] + '.mp4'
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return output_path
