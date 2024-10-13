import torch
import torch.nn as nn
from torchvision import transforms
import transformers
from tools.cvt import CvT
from tools.encoder_projection import EncoderPermuteProject

class CvT2DistilGPT2MIMICXRChenInference(nn.Module):
    def __init__(self, ckpt_path, ckpt_zoo_dir):
        super().__init__()
        
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.decoder_max_len = 512  # Adjust this value if needed

        # Initialize model components
        self.initialize_model()

        # Load the trained weights
        self.load_trained_weights(ckpt_path)

        # Tokenizer
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(
            f'{self.ckpt_zoo_dir}/distilgpt2',
            local_files_only=True,
        )
        self.tokenizer.add_special_tokens({"bos_token": "[BOS]", 'pad_token': '[PAD]'})

        # Image transformations
        self.transforms = transforms.Compose([
            transforms.Resize(size=384),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def initialize_model(self):
        # Encoder
        self.encoder = CvT(
            warm_start=False,
            model_config='cvt-21-384x384',
            ckpt_name='CvT-21-384x384-IN-22k',
            ckpt_dir=self.ckpt_zoo_dir,
            is_encoder=True,
        )
        self.encoder_projection = EncoderPermuteProject(
            permute_encoder_last_hidden_state=[0, 2, 1],
            encoder_last_hidden_state_size=384,
            decoder_hidden_state_size=768,
        )

        # Decoder
        config = transformers.GPT2Config.from_pretrained(
            f'{self.ckpt_zoo_dir}/distilgpt2',
            local_files_only=True,
        )
        config.add_cross_attention = True
        config.is_decoder = True
        decoder = transformers.GPT2LMHeadModel(config)
        decoder.resize_token_embeddings(config.vocab_size + 2)

        # EncoderDecoderModel
        class DummyEncoder:
            main_input_name = 'dummy'
            class DummyConfig(transformers.PretrainedConfig):
                model_type = 'bert'
            config = DummyConfig()
            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size
            def get_output_embeddings(cls):
                return None
            def forward(self):
                return None

        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)
        self.decoder = transformers.EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

    def load_trained_weights(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        self.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder')}, strict=False)
        self.encoder_projection.load_state_dict({k.replace('encoder_projection.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder_projection')})
        self.decoder.load_state_dict({k.replace('decoder.encoder_decoder.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('decoder.encoder_decoder')}, strict=False)

    def encoder_forward(self, images):
        image_features = self.encoder(images)['last_hidden_state']
        image_features = self.encoder_projection(image_features)['projected_encoder_last_hidden_state']
        return transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)

    def generate(self, images, num_beams=4):
        with torch.no_grad():
            images = self.transforms(images).unsqueeze(0)
            encoder_outputs = self.encoder_forward(images)

            outputs = self.decoder.generate(
                max_length=self.decoder_max_len,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=num_beams,
                return_dict_in_generate=True,
                use_cache=True,
                encoder_outputs=encoder_outputs,
            )

            generated_text = self.tokenizer.decode(outputs['sequences'][0], skip_special_tokens=True)
            return generated_text

# # Usage example:
# model = CvT2DistilGPT2MIMICXRChenInference("./checkpoints/epoch=8-val_chen_cider=0.425092.ckpt", "./checkpoints/")
# from PIL import Image
# image = Image.open("./sample_data/1.jpeg").convert('RGB')
# generated_report = model.generate(image)
# print(generated_report)