import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer, SwinModel

class R2GenGPT(nn.Module):
    """
    Specifically for 0 shot inference
    """
    def __init__(self):
        super().__init__()
        self.vision_model_name = 'microsoft/swin-base-patch4-window7-224'
        # self.llama_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
        self.llama_model_name = 'meta-llama/Llama-2-7b-chat-hf'
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        
        # Vision encoder
        print(f'Loading vision model.. {self.vision_model_name}')
        self.visual_encoder = SwinModel.from_pretrained(self.vision_model_name).half()  # Convert to half
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        print('Vision model loaded')

        # Text model parts
        print(f'Loading LLaMA model.. {self.llama_model_name}')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.llama_model_name, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        
        self.llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_model_name,
            torch_dtype=torch.float16,
        )
        self.embed_tokens = self.llama_model.get_input_embeddings()
        for param in self.llama_model.parameters():
            param.requires_grad = False
        print('LLaMA loaded for inference')

        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)

    def encode_img(self, image):
        """
        Preprocess image for vision model and for llama input
        """
        device = image.device
        dtype = self.llama_model.dtype  # Get model's dtype

        image_embed = self.visual_encoder(image)['last_hidden_state'].to(device, dtype=dtype)
        inputs_llama = self.llama_proj(image_embed)
        inputs_llama = inputs_llama.to(dtype)  # Ensure projection output matches dtype

        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        """Wrap image embeddings with prompt."""
        # Prompt
        prompt = f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def decode(self, output_token):
        """Decode output tokens to text."""
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
            
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    @torch.no_grad()
    def generate(self, image_tensor):
        """
        Function to generate the radiology report
        """
        # Set to eval mode
        self.eval()
        
        # Encode image
        img_embeds, atts_img = self.encode_img(image_tensor)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        # Preprocess
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1], dtype=atts_img.dtype, device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        # Concat image and text embeddings
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        # For attention embeddings
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        # Generate
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=3,
            do_sample=False,
            min_new_tokens=80,
            max_new_tokens=120,
            repetition_penalty=2.0,
            length_penalty=2.0,
            temperature=0,
        )
        
        report = self.decode(outputs[0])
        return report

    @classmethod
    def from_pretrained(cls, checkpoint_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        model = cls()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model = model.to(device)
        model.eval()
        
        return model