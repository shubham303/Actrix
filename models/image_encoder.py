import torch
import torch.nn as nn
from PIL import Image
import numpy as np


class ImageEncoder(nn.Module):
    """
    Generic image encoder that supports different backbone models.
    Takes an image and returns a single embedding vector using a specified pooling strategy.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        pooling_strategy: str = "mean",
        output_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the image encoder.
        
        Args:
            backbone: Backbone model that outputs token embeddings
            pooling_strategy: Strategy to combine token embeddings ("mean", "cls", "max")
            output_dim: If provided, project embeddings to this dimension
            device: Device to run the model on
        """
        super().__init__()
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set backbone and get embedding dimension
        self.backbone = backbone
        self.embed_dim = getattr(backbone, "get_embedding_dim", lambda: backbone.embed_dim)()
        
        # Set pooling strategy
        self.pooling_strategy = pooling_strategy
        
        # Initialize projection layer if needed
        self.output_dim = output_dim
        if output_dim is not None:
            self.projection = nn.Linear(self.embed_dim, output_dim).to(self.device)
        else:
            self.projection = None
    
    def _pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply pooling strategy to token embeddings"""
        if self.pooling_strategy == "mean":
            # Mean pooling across tokens
            return embeddings.mean(dim=1)
        elif self.pooling_strategy == "cls":
            # Use CLS token embedding (first token)
            return embeddings[:, 0]
        elif self.pooling_strategy == "max":
            # Max pooling across tokens
            return embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Pooling strategy {self.pooling_strategy} not supported")
    
    def forward(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Extract a single embedding vector from an image
        
        Args:
            image: Input image (format depends on backbone model)
            
        Returns:
            Embedding tensor of shape (1, embed_dim) or (1, output_dim)
        """
        # Get token embeddings from backbone
        token_embeddings = self.backbone(image)
        
        # Apply pooling to get a single embedding vector per image
        pooled_embedding = self._pool_embeddings(token_embeddings)
        
        # Apply projection if needed
        if self.projection is not None:
            pooled_embedding = self.projection(pooled_embedding)
        
        return pooled_embedding
    
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension"""
        return self.output_dim if self.output_dim is not None else self.embed_dim
    