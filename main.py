import torch
import triton
import triton.language as tl
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Define Triton kernel for image convolution
@triton.jit
def convolution_kernel(
    # Pointers to matrices
    input_ptr, kernel_ptr, output_ptr,
    # Matrix dimensions
    batch_size, height, width, channels, kernel_size,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Compute indices
    pid = tl.program_id(0)
    batch_idx = pid // (height * width)
    
    # Compute the row and column indices of the output matrix
    row_idx = (pid % (height * width)) // width
    col_idx = (pid % (height * width)) % width
    
    # Do not compute if we're out of bounds
    if row_idx >= height or col_idx >= width or batch_idx >= batch_size:
        return
    
    # Compute the half width of the kernel
    half_kernel = kernel_size // 2
    
    # Initialize the accumulator
    acc = tl.zeros([channels], dtype=tl.float32)
    
    # Loop over kernel rows and columns
    for k_row in range(kernel_size):
        for k_col in range(kernel_size):
            # Compute the input row and column indices
            in_row = row_idx + k_row - half_kernel
            in_col = col_idx + k_col - half_kernel
            
            # Check boundary conditions
            if 0 <= in_row < height and 0 <= in_col < width:
                # Load the input value and kernel value
                input_offset = batch_idx * height * width * channels + in_row * width * channels + in_col * channels
                kernel_offset = k_row * kernel_size + k_col
                
                # Load input and kernel values
                input_values = tl.load(input_ptr + input_offset, mask=True, other=0.0)
                kernel_value = tl.load(kernel_ptr + kernel_offset)
                
                # Accumulate
                acc += input_values * kernel_value
    
    # Write the result
    output_offset = batch_idx * height * width * channels + row_idx * width * channels + col_idx * channels
    tl.store(output_ptr + output_offset, acc)


class TritonImageProcessor:
    def __init__(self):
        """Initialize the Triton Image Processor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Operations will run on CPU.")
    
    def load_image(self, path):
        """Load an image and convert to tensor."""
        img = Image.open(path).convert('RGB')
        img_np = np.array(img) / 255.0
        img_tensor = torch.tensor(img_np, device=self.device, dtype=torch.float32)
        return img_tensor
    
    def save_image(self, tensor, path):
        """Save a tensor as an image."""
        if tensor.device != torch.device('cpu'):
            tensor = tensor.cpu()
        img_np = (tensor.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        img.save(path)
    
    def apply_kernel(self, image, kernel):
        """Apply a convolution kernel to an image using Triton."""
        # Check dimensions
        if len(image.shape) == 3:  # Single image
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Get dimensions
        batch_size, height, width, channels = image.shape
        kernel_size = kernel.shape[0]
        
        # Prepare output tensor
        output = torch.zeros_like(image)
        
        # Define grid
        grid = (batch_size * height * width,)
        
        # Launch kernel
        convolution_kernel[grid](
            image.contiguous(), kernel.contiguous(), output,
            batch_size, height, width, channels, kernel_size,
            BLOCK_SIZE_M=16, BLOCK_SIZE_N=16
        )
        
        return output.squeeze(0) if batch_size == 1 else output
    
    def gaussian_blur(self, image, sigma=1.0):
        """Apply Gaussian blur to an image."""
        # Create Gaussian kernel
        kernel_size = max(3, int(2 * sigma) * 2 + 1)
        kernel = torch.zeros((kernel_size, kernel_size), device=self.device)
        center = kernel_size // 2
        
        # Fill kernel with Gaussian values
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        return self.apply_kernel(image, kernel)
    
    def sobel_edge_detection(self, image):
        """Apply Sobel edge detection to an image."""
        # Define Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], device=self.device, dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], device=self.device, dtype=torch.float32)
        
        # Apply kernels
        grad_x = self.apply_kernel(image, sobel_x)
        grad_y = self.apply_kernel(image, sobel_y)
        
        # Compute gradient magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        edges = edges / edges.max()
        
        return edges
    
    def sharpen(self, image, strength=1.0):
        """Sharpen an image."""
        # Create sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], device=self.device, dtype=torch.float32)
        
        return self.apply_kernel(image, kernel)


# Custom Triton implementation of image resizing
@triton.jit
def resize_kernel(
    input_ptr, output_ptr,
    in_height, in_width, channels,
    out_height, out_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute output pixel position
    pid = tl.program_id(0)
    out_row = pid // out_width
    out_col = pid % out_width
    
    # Skip if out of bounds
    if out_row >= out_height or out_col >= out_width:
        return
    
    # Compute scaling factors
    scale_y = in_height / out_height
    scale_x = in_width / out_width
    
    # Compute corresponding input pixel position (using bilinear interpolation)
    in_y = out_row * scale_y
    in_x = out_col * scale_x
    
    # Get four surrounding pixel positions
    in_y0 = int(in_y)
    in_x0 = int(in_x)
    in_y1 = min(in_y0 + 1, in_height - 1)
    in_x1 = min(in_x0 + 1, in_width - 1)
    
    # Compute interpolation weights
    wy1 = in_y - in_y0
    wx1 = in_x - in_x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1
    
    # Load the four surrounding pixels
    top_left_offset = in_y0 * in_width * channels + in_x0 * channels
    top_right_offset = in_y0 * in_width * channels + in_x1 * channels
    bottom_left_offset = in_y1 * in_width * channels + in_x0 * channels
    bottom_right_offset = in_y1 * in_width * channels + in_x1 * channels
    
    top_left = tl.load(input_ptr + top_left_offset, mask=True, other=0.0)
    top_right = tl.load(input_ptr + top_right_offset, mask=True, other=0.0)
    bottom_left = tl.load(input_ptr + bottom_left_offset, mask=True, other=0.0)
    bottom_right = tl.load(input_ptr + bottom_right_offset, mask=True, other=0.0)
    
    # Bilinear interpolation
    result = (top_left * (wx0 * wy0) +
              top_right * (wx1 * wy0) +
              bottom_left * (wx0 * wy1) +
              bottom_right * (wx1 * wy1))
    
    # Store result
    output_offset = out_row * out_width * channels + out_col * channels
    tl.store(output_ptr + output_offset, result)


# Add resize method to TritonImageProcessor
def resize(self, image, new_height, new_width):
    """Resize an image using Triton."""
    # Ensure image is a 4D tensor (batch, height, width, channels)
    if len(image.shape) == 3:  # Single image
        image = image.unsqueeze(0)  # Add batch dimension
    
    batch_size, height, width, channels = image.shape
    
    # Prepare output tensor
    output = torch.zeros((batch_size, new_height, new_width, channels), 
                          device=self.device, dtype=torch.float32)
    
    # Process each image in the batch
    for b in range(batch_size):
        # Define grid
        grid = (new_height * new_width,)
        
        # Launch kernel
        resize_kernel[grid](
            image[b].contiguous(), output[b].contiguous(),
            height, width, channels,
            new_height, new_width,
            BLOCK_SIZE=16
        )
    
    return output.squeeze(0) if batch_size == 1 else output

# Add the resize method to the class
TritonImageProcessor.resize = resize


def benchmark(func, *args, **kwargs):
    """Benchmark a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Demo usage
if __name__ == "__main__":
    # Create processor
    processor = TritonImageProcessor()
    
    # Load image
    img = processor.load_image("input.jpg")
    print(f"Image loaded with shape: {img.shape}")
    
    # Benchmark operations
    print("Benchmarking operations...")
    
    # Blur
    blurred, blur_time = benchmark(processor.gaussian_blur, img, sigma=2.0)
    print(f"Gaussian blur completed in {blur_time:.4f} seconds")
    
    # Edge detection
    edges, edge_time = benchmark(processor.sobel_edge_detection, img)
    print(f"Edge detection completed in {edge_time:.4f} seconds")
    
    # Sharpen
    sharpened, sharpen_time = benchmark(processor.sharpen, img, strength=1.5)
    print(f"Sharpening completed in {sharpen_time:.4f} seconds")
    
    # Resize
    resized, resize_time = benchmark(processor.resize, img, 512, 512)
    print(f"Resizing completed in {resize_time:.4f} seconds")
    
    # Save results
    processor.save_image(blurred, "blurred.jpg")
    processor.save_image(edges, "edges.jpg")
    processor.save_image(sharpened, "sharpened.jpg")
    processor.save_image(resized, "resized.jpg")
    
    print("Results saved!")
    
    # Compare with PyTorch (CPU and GPU)
    print("\nComparing with PyTorch implementations...")
    
    # PyTorch GPU implementation of Gaussian blur
    def torch_gaussian_blur(image, kernel_size=5, sigma=2.0):
        # Create Gaussian kernel
        kernel = torch.zeros((kernel_size, kernel_size), device=image.device)
        center = kernel_size // 2
        
        # Fill kernel with Gaussian values
        for i in range(kernel_size):
            for j in range(kernel_size):
                x, y = i - center, j - center
                kernel[i, j] = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Reshape for torch.nn.functional.conv2d
        kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(3, 1, 1, 1)
        
        # Prepare image for convolution
        img = image.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
        
        # Apply convolution
        import torch.nn.functional as F
        output = F.conv2d(img, kernel, padding=center, groups=3)
        
        # Reshape back
        return output.squeeze(0).permute(1, 2, 0)
    
    # Benchmark PyTorch implementation
    torch_blurred, torch_time = benchmark(torch_gaussian_blur, img, kernel_size=5, sigma=2.0)
    print(f"PyTorch Gaussian blur completed in {torch_time:.4f} seconds")
    
    # Calculate speedup
    speedup = torch_time / blur_time
    print(f"Triton implementation is {speedup:.2f}x faster than PyTorch")
