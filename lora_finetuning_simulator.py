import numpy as np
import logging

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("LoRAFineTuning")

class LoRALayerSimulation:
    """
    Simulates Low-Rank Adaptation (LoRA) on a neural network weight layer.
    Demonstrates parameter freezing, low-rank matrix decomposition (A and B),
    and forward pass merging (W_final = W_base + (alpha/r) * (B * A)).
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 16.0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # 1. Base Model Weight Matrix (Frozen - simulating pretrained LLM weights)
        np.random.seed(42)
        self.W_base = np.random.randn(out_features, in_features)
        
        # 2. LoRA Adapter Matrices: A and B
        # Matrix A is initialized with Gaussian distribution
        self.A = np.random.randn(rank, in_features)
        # Matrix B is initialized with zeros (so initial adapter effect is zero)
        self.B = np.zeros((out_features, rank))

        logger.info(
            f"LoRALayer initialized | Base Weights Shape: {self.W_base.shape} | "
            f"Rank r: {rank} | Matrix A: {self.A.shape} | Matrix B: {self.B.shape}"
        )

    def count_parameters(self) -> dict:
        """Calculates total trainable vs frozen parameters to show parameter efficiency."""
        frozen_params = self.W_base.size
        trainable_params = self.A.size + self.B.size
        return {
            "frozen_params": frozen_params,
            "trainable_params": trainable_params,
            "reduction_percentage": (1 - (trainable_params / frozen_params)) * 100
        }

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward Pass:
        y = (W_base * x) + scaling * (B * A * x)
        """
        # Frozen base weight computation
        base_output = np.dot(x, self.W_base.T)
        
        # Adapter computation (Low-Rank Update)
        lora_adapter_output = np.dot(x, self.A.T) # Compress to rank r
        lora_adapter_output = np.dot(lora_adapter_output, self.B.T) # Expand to out_features
        
        # Combined output with scaling factor
        final_output = base_output + (self.scaling * lora_adapter_output)
        return final_output

if __name__ == "__main__":
    print("\n" + "="*65)
    print("PEFT & LoRA (LOW-RANK ADAPTATION) ARCHITECTURE SIMULATOR")
    print("="*65 + "\n")

    # Dimensions simulating LLM Attention Projection Layer (e.g., in_features=4096, out_features=4096)
    in_dim = 4096
    out_dim = 4096
    rank_r = 8
    lora_alpha = 16.0

    # Initialize LoRA Layer
    lora_layer = LoRALayerSimulation(in_features=in_dim, out_features=out_dim, rank=rank_r, alpha=lora_alpha)

    # Calculate parameter stats
    param_stats = lora_layer.count_parameters()
    print("--- PARAMETER EFFICIENCY COMPARISON ---")
    print(f"Base Frozen Parameters   : {param_stats['frozen_params']:,}")
    print(f"LoRA Trainable Parameters: {param_stats['trainable_params']:,}")
    print(f"Parameter Reduction      : {param_stats['reduction_percentage']:.2f}% fewer parameters to train!")

    print("\n" + "-"*65 + "\n")

    # Simulate Forward Pass with mock input tensor
    print("--- FORWARD PASS SIMULATION ---")
    mock_input_tokens = np.random.randn(2, in_dim) # Batch size=2, Sequence dim=4096
    output = lora_layer.forward(mock_input_tokens)
    
    print(f"Input Tensor Shape : {mock_input_tokens.shape}")
    print(f"Output Tensor Shape: {output.shape}")
    print("Forward pass completed successfully with merged frozen + adapter weights.")