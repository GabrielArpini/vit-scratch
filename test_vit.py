import torch
from vit import ViT

# Test parameters
batch_size = 2
channels = 3
height = 224
width = 224
patch_size = 16
d_model = 512
num_classes = 10

print("Testing ViT Implementation")

# Create model
print("Creating ViT model...")
model = ViT(
    P=patch_size,
    C=channels,
    d_model=d_model,
    H=height,
    W=width,
    num_classes=num_classes
)
print("Model created successfully")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Create random input
print("Creating random input...")
x = torch.randn(batch_size, channels, height, width)
print(f"Input shape: {x.shape}")

# Forward pass
print("Running forward pass...")
output = model(x)
print(f"Output shape: {output.shape}")
print(f"Expected shape: ({batch_size}, {num_classes})")

# Check output shape
assert output.shape == (batch_size, num_classes), f"Wrong output shape! Got {output.shape}, expected ({batch_size}, {num_classes})"
print("Output shape is correct")

# Test backward pass
print("Testing backward pass...")
loss = output.sum()
loss.backward()
print("Backward pass successful")

# Check gradients
print("Checking gradients...")
has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
assert has_gradients, "No gradients computed!"
print("Gradients computed successfully")

# Test with different batch size
print("Testing with different batch size...")
x_single = torch.randn(1, channels, height, width)
output_single = model(x_single)
assert output_single.shape == (1, num_classes), f"Wrong shape for single batch! Got {output_single.shape}"
print(f"Single batch output shape: {output_single.shape}")
print("Works with different batch sizes")

print("All tests passed!")
