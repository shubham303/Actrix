import torch
import torch.nn as nn
from typing import Union, Optional, Dict, List, Callable
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, AutoModel

class DINOv2Encoder(nn.Module):
    """
    Encoder specifically for DINOv2 models from HuggingFace Transformers.
    Takes an image and returns the raw embedding from DINOv2.
    """
    
    MODEL_VARIANTS = {
        "small": {
            "model_id": "facebook/dinov2-small",
            "embed_dim": 384
        },
        "base": {
            "model_id": "facebook/dinov2-base", 
            "embed_dim": 768
        },
        "large": {
            "model_id": "facebook/dinov2-large",
            "embed_dim": 1024
        },
        "giant": {
            "model_id": "facebook/dinov2-giant",
            "embed_dim": 1536
        }
    }
    
    def __init__(
        self, 
        variant: str = "base",
        device: Optional[torch.device] = None,
        use_traced_model: bool = False
    ):
        """
        Initialize the DINOv2 encoder.
        
        Args:
            variant: DINOv2 variant ("small", "base", "large", "giant")
            device: Device to load the model on, defaults to cuda if available
            use_traced_model: Whether to use torch.jit.trace for faster inference
        """
        super().__init__()
        
        if variant not in self.MODEL_VARIANTS:
            raise ValueError(f"Variant {variant} not supported. Available variants: {list(self.MODEL_VARIANTS.keys())}")
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model info
        self.variant = variant
        self.model_info = self.MODEL_VARIANTS[variant]
        self.embed_dim = self.model_info["embed_dim"]
        
        # Load model and processor
        print(f"Loading DINOv2 {variant} from {self.model_info['model_id']}...")
        self.processor = AutoImageProcessor.from_pretrained(self.model_info["model_id"])
        self.model = AutoModel.from_pretrained(self.model_info["model_id"]).to(self.device)
        self.model.eval()
        
        # Create traced model if requested
        self.use_traced_model = use_traced_model
        self.traced_model = None
        if use_traced_model:
            self._create_traced_model()
    
    def _create_traced_model(self):
        """Create a traced version of the model for faster inference"""
        print("Creating traced model...")
        # Turn off return_dict for tracing
        self.model.config.return_dict = False
        
        # Create a dummy input for tracing
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Trace the model
        with torch.no_grad():
            self.traced_model = torch.jit.trace(self.model, [dummy_input])
    
    def preprocess(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Process the image into model input format"""
        if isinstance(image, str):
            # Load image from file path
            image = Image.open(image).convert("RGB")
        
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            image = Image.fromarray(image.astype(np.uint8))
        
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.ndim == 3 and image.shape[0] == 3:
                # CxHxW format
                image = image.permute(1, 2, 0).cpu().numpy()
                image = Image.fromarray((image * 255).astype(np.uint8))
            elif image.ndim == 3 and image.shape[2] == 3:
                # HxWxC format
                image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            else:
                raise ValueError(f"Tensor image must have shape (3, H, W) or (H, W, 3), got {image.shape}")
        
        # Process using the HuggingFace processor
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def forward(self, image: Union[str, Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Extract token embeddings from an image
        
        Args:
            image: Input image (file path, PIL Image, numpy array, or tensor)
            
        Returns:
            Token embeddings tensor of shape (1, num_tokens, embed_dim)
        """
        # Preprocess image
        inputs = self.preprocess(image)
        
        # Extract features
        with torch.no_grad():
            if self.use_traced_model and self.traced_model is not None:
                outputs = self.traced_model(inputs["pixel_values"])
                token_embeddings = outputs[0]  # [batch_size, num_tokens, embed_dim]
            else:
                outputs = self.model(**inputs)
                token_embeddings = outputs.last_hidden_state  # [batch_size, num_tokens, embed_dim]
        
        return token_embeddings
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension"""
        return self.embed_dim


