import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import torch
from sklearn.cluster import DBSCAN
from PIL import Image
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity

import dotenv

dotenv.load_dotenv()

def extract_frames(video_path, fps=8, max_duration=10):
    """Extract frames from video at specified FPS, limiting to max duration."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / orig_fps
    
    # Limit to max_duration
    actual_duration = min(duration, max_duration)
    max_frames = int(actual_duration * fps)
    
    # Calculate frame interval
    interval = orig_fps / fps
    
    frames = []
    count = 0
    frame_idx = 0
    
    while count < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 224x224
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)
        
        count += 1
        frame_idx += interval
    
    cap.release()
    return frames

def get_embeddings(frames, model, processor, device):
    """Get DINOv2 embeddings for all frames."""
    all_embeddings = []
    
    for frame in frames:
        # Convert to PIL image
        pil_image = Image.fromarray(frame)

        inputs = processor(images=pil_image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get embeddings (exclude CLS token)
        embeddings = outputs.last_hidden_state.squeeze().cpu().numpy()
        all_embeddings.append(embeddings[1:, :])  # Exclude CLS token
    
    return all_embeddings

def reshape_embeddings_to_grid(all_embeddings):
    """Reshape embeddings to grid format with spatial dimensions."""
    num_frames = len(all_embeddings)
    num_patches = all_embeddings[0].shape[0]  # 256 patches
    patch_size = int(np.sqrt(num_patches))  # Should be 16 for 16x16 grid
    embedding_dim = all_embeddings[0].shape[1]
    
    # Create a 4D array: [batch(frames), height, width, embedding_dim]
    reshaped_embeddings = np.zeros((num_frames, patch_size, patch_size, embedding_dim))
    
    for frame_idx in range(num_frames):
        frame_embeddings = all_embeddings[frame_idx]
        # Reshape from [256, dim] to [16, 16, dim]
        reshaped_embeddings[frame_idx] = frame_embeddings.reshape(patch_size, patch_size, embedding_dim)
    
    return reshaped_embeddings

def cluster_combined_patches(reshaped_embeddings, eps=0.2, min_samples=1):
    """Cluster embeddings for each 2x2 patch neighborhood across frames."""
    num_frames, h, w, embedding_dim = reshaped_embeddings.shape
    # New dimensions after combining 2x2 patches
    new_h, new_w = h//2, w//2
    
    # Initialize storage for reduced embeddings
    reduced_embeddings = np.zeros((new_h * new_w, embedding_dim))
    total_vectors = num_frames * new_h * new_w * 4  # Each combined patch has 4 original patches
    remaining_vectors = 0
    
    # Process each 2x2 patch neighborhood
    for i in range(new_h):
        for j in range(new_w):
            # Collect embeddings for this 2x2 region from all frames
            combined_embeddings = []
            
            for frame_idx in range(num_frames):
                # Extract the 2x2 patch region
                patch_2x2 = reshaped_embeddings[frame_idx, i*2:(i+1)*2, j*2:(j+1)*2, :]
                # Reshape to 4 separate patch vectors
                patches = patch_2x2.reshape(-1, embedding_dim)
                # Add all 4 patches from this frame
                combined_embeddings.extend(patches)
            
            # Convert to numpy array
            combined_embeddings = np.array(combined_embeddings)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(combined_embeddings)
            labels = clustering.labels_
            
            # Count clusters (excluding noise points with label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # For each cluster, compute the mean embedding as representative
            cluster_means = []
            for cluster_idx in range(n_clusters):
                cluster_points = combined_embeddings[labels == cluster_idx]
                cluster_mean = np.mean(cluster_points, axis=0)
                cluster_means.append(cluster_mean)
            
            # Handle noise points (label -1) by keeping them as is
            noise_points = combined_embeddings[labels == -1]
            
            # Store results - use mean of cluster means
            if cluster_means:
                reduced_embeddings[i*new_w + j] = np.mean(cluster_means, axis=0)
            elif len(noise_points) > 0:
                reduced_embeddings[i*new_w + j] = np.mean(noise_points, axis=0)
            else:
                reduced_embeddings[i*new_w + j] = np.zeros(embedding_dim)
            
            # Track number of vectors after reduction
            remaining_vectors += n_clusters + len(noise_points)
    
    # Return reduced embeddings and reduction statistics
    return reduced_embeddings, total_vectors, remaining_vectors

def main(video_path, eps=0.2, min_samples=1):
    """Main function to process video, extract embeddings, and apply clustering."""
    print(f"Processing video: {video_path}")
    start_time = time.time()

    device = torch.device("mps")
    
    # Step 1: Extract frames
    frames = extract_frames(video_path)
    print(f"Extracted {len(frames)} frames")
    
    # Step 2: Load DINOv2 model
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)
    
    # Step 3: Get embeddings for all frames
    all_embeddings = get_embeddings(frames, model, processor, device)
    
    # Step 4: Reshape embeddings to grid format
    reshaped_embeddings = reshape_embeddings_to_grid(all_embeddings)
    
    # Step 5: Cluster embeddings for each 2x2 patch neighborhood
    reduced_embeddings, total_vectors, remaining_vectors = cluster_combined_patches(
        reshaped_embeddings, eps=eps, min_samples=min_samples
    )
    
    # Calculate reduction percentage
    reduction_pct = 100 * (1 - remaining_vectors / total_vectors)
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Total embedding vectors before reduction: {total_vectors}")
    print(f"Total embedding vectors after reduction: {remaining_vectors}")
    print(f"Reduction achieved: {reduction_pct:.2f}%")
    
    return reduced_embeddings, frames

if __name__ == "__main__":
    import video_utils
    video_path = "/Users/shubhamrandive/Documents/codes/Actrix/data/hmdb51/hmdb51_org/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi"

    # Parameters for DBSCAN clustering
    eps = 0.1  # Maximum distance between samples in a cluster
    min_samples = 1  # Minimum samples in a neighborhood to form a core point
    
    reduced_embeddings, frames = main(video_path, eps=eps, min_samples=min_samples)
    
    # Visualize first frame for reference
    plt.figure(figsize=(8, 8))
    plt.imshow(frames[0])
    plt.title("First frame of processed video")
    plt.axis('off')
    plt.show()