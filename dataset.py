import os
import torch
from torch.utils.data import Dataset
from decord import VideoReader
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional, Union


class VideoDataset(Dataset):
    def __init__(
        self,
        folder_paths: List[str],
        max_clip_duration: float = 10.0,
        frame_rate: Optional[int] = None,
        transform=None,
        extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv'),
    ):
        """
        Generic video dataset for pretraining that loads videos recursively from folders.
        
        Args:
            folder_paths: List of folder paths to search for videos
            max_clip_duration: Maximum duration of video clips in seconds
            frame_rate: Target frame rate (if None, uses original video frame rate)
            transform: Optional transforms to apply to the video frames
            extensions: Tuple of valid video file extensions
        """
        self.max_clip_duration = max_clip_duration
        self.frame_rate = frame_rate
        self.transform = transform
        self.extensions = extensions
        
        # Collect all video paths recursively
        self.video_paths = []
        for folder in folder_paths:
            self._collect_videos(folder)
            
    def _collect_videos(self, folder_path: str) -> None:
        """Collect video paths recursively from the given folder"""
        for root, _, files in os.walk(folder_path):
            for file in tqdm(files, desc=f"Collecting videos from {os.path.basename(folder_path)}"):
                if file.lower().endswith(self.extensions):
                    self.video_paths.append(os.path.join(root, file))
                    
    def __len__(self) -> int:
        return len(self.video_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        video_path = self.video_paths[idx]
        
        # Load video using decord
        vr = VideoReader(video_path)
        
        # Calculate number of frames to sample
        original_fps = vr.get_avg_fps()
        target_fps = original_fps if self.frame_rate is None else self.frame_rate
        max_frames = int(self.max_clip_duration * target_fps)
        
        # If video is longer than max_clip_duration, select a random segment
        num_frames = len(vr)
        
        if num_frames > max_frames:
            # Choose a random starting point
            start_idx = np.random.randint(0, num_frames - max_frames + 1)
            frame_indices = list(range(start_idx, start_idx + max_frames))
        else:
            # Use all frames if the video is shorter than max_frames
            frame_indices = list(range(num_frames))
        
        # Sample frames
        video_frames = vr.get_batch(frame_indices).asnumpy()
        
        # Convert to torch tensor with THWC format (Time, Height, Width, Channels)
        video_tensor = torch.from_numpy(video_frames)
        
        # Apply transform if specified
        if self.transform:
            video_tensor = self.transform(video_tensor)
            
        return video_tensor