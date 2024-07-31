import torch
import huggingface_hub
from dataset_smapler import GenerationDataset
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image, Transformer2DModel, PixArtSigmaPipeline, StableDiffusion3Pipeline, DiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader
import os
import yaml

with open(r"text2image/t2i_configs.yml") as file:
    params_list = yaml.load(file, Loader=yaml.FullLoader)

# huggingface_hub.login(token='YOUR HF ACCESS TOKEN')
    
class InitializeModels:
    def __init__(self, model_name, outputdir_name):
        self.model_path = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        dir_name = model_name.replace('/', '_')

        self.dir  = f"{outputdir_name}/{dir_name}"
        if not os.path.exists(self.dir):
            os.mkdir(self.dir)
  
        self.pipe_def()


    def pipe_def(self):

        if self.model_path == 'stabilityai/stable-diffusion-3-medium':
            self.pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, 
                                                                 torch_dtype=torch.float16, 
                                                                 variant="fp16")
        
        elif self.model_path == 'stabilityai/stable-diffusion-xl-base-1.0' or  self.model_path == 'playgroundai/playground-v2.5-1024px-aesthetic':
            self.pipe == DiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)

        elif self.model_path == 'runwayml/stable-diffusion-v1-5' or self.model_path == 'prompthero/openjourney':
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)

        elif self.model_path == 'stabilityai/sdxl-turbo' or self.model_path == 'kandinsky-community/kandinsky-3':
            self.pipe = AutoPipelineForText2Image.from_pretrained(self.model_path, torch_dtype=torch.float16, variant="fp16")

        elif self.model_path == 'PixArt-alpha/PixArt-Sigma-XL-2-512-MS':
            self.pipe = PixArtSigmaPipeline.from_pretrained(
                        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        )
            
        elif self.model_path == "DeepFloyd/IF-I-XL-v1.0":
            self.pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
            self.pipe.enable_model_cpu_offload()

        self.pipe.to(self.device)


    def generate_image(self, batch, ids):
        images = self.pipe(batch, 
                            num_inference_steps=28,
                            guidance_scale=7.0,).images
        
        for name, pic in zip(batch, images):
            idx = ids[name]
            pic.save(f"{self.dir}/img_{idx}.png")


if __name__ == '__main__':
    our_set = GenerationDataset(params_list["DATASET_PATH"][0])
    loader = DataLoader(our_set, batch_size=16)

    print(params_list)

    model = InitializeModels(params_list["MODEL_NAME"][0], params_list["OUTPUT_DIR"][0])

    for batch in loader:
        model.generate_image(batch, our_set.ids)
