import os
import json
import time
from tqdm import tqdm
from PIL import Image

import torch
import torchvision.transforms as transforms

from models.R2GenGPT import R2GenGPT


class TestInference:
    def __init__(self, checkpoint_path='saved_models/checkpoint_epoch3_step541579_bleu0.109445_cider0.171724.pth',
                 input_dir='./sample_data',
                 output_dir='./inference_results'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        print(f"Using device: {self.device}")
        self.model = R2GenGPT.from_pretrained(checkpoint_path, device=self.device)
        print("Model loaded successfully")
        
    def process_single_image(self, image_path):
        """Process a single image and return its report."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generate report
            report = self.model.generate(image_tensor)
            return report
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return f"Error: {str(e)}"
    
    def run_inference(self):
        """Run inference on all images in input directory."""
        # Get all image files
        image_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        # Dictionary to store results
        results = {}
        total_time = 0
        
        # Process each image
        for img_file in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(self.input_dir, img_file)
            
            # Time the inference
            start_time = time.time()
            report = self.process_single_image(img_path)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # Store results
            results[img_file] = {
                "report": report,
                "inference_time": f"{inference_time:.2f} seconds"
            }
            
            # Print individual result
            print(f"\nResults for {img_file}:")
            print("-" * 50)
            print(f"Report: {report}")
            print(f"Inference time: {inference_time:.2f} seconds")
            print("-" * 50)
        
        # Calculate and print statistics
        avg_time = total_time / len(image_files)
        stats = {
            "total_images": len(image_files),
            "total_time": f"{total_time:.2f} seconds",
            "average_time": f"{avg_time:.2f} seconds"
        }
        
        # Save results to JSON
        output_file = os.path.join(self.output_dir, "inference_results.json")
        with open(output_file, 'w') as f:
            json.dump({
                "statistics": stats,
                "results": results
            }, f, indent=4)
            
        print("\nInference Summary:")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Total time: {stats['total_time']}")
        print(f"Average time per image: {stats['average_time']}")
        print(f"Results saved to: {output_file}")

def main():
    # Set up paths
    checkpoint_path = 'saved_models/checkpoint_epoch3_step541579_bleu0.109445_cider0.171724.pth'
    input_dir = './sample_data/p10999737/s52341872/62973129-7b40a2cb-1e8778aa-89086ca9-88ff3978.jpg'  # Directory containing test images
    output_dir = './inference_results'  # Directory to save results
    
    # Create and run test
    tester = TestInference(
        checkpoint_path=checkpoint_path,
        input_dir=input_dir,
        output_dir=output_dir
    )
    tester.run_inference()

if __name__ == "__main__":
    main()
