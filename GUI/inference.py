import torch
import torchvision.transforms as transforms
from PIL import Image
from models.R2GenGPT import R2GenGPT

class XrayReportGenerator:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, checkpoint_path='saved_models/checkpoint_epoch3_step541579_bleu0.109445_cider0.171724.pth'):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        torch.cuda.empty_cache()
        self.model = R2GenGPT.from_pretrained(checkpoint_path, device=self.device)
        self.model = self.model.half()

        self._initialized = True
        print("Model initialized successfully")

    def generate_report(self, image_file):
        try:
            image = Image.open(image_file).convert('RGB')
            # image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)
            report = self.model.generate(image_tensor)
            return report
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}"

generator = XrayReportGenerator()