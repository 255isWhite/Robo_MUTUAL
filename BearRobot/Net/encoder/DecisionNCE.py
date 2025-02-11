import DecisionNCE
import torch
import torch.nn as nn

   
class DecisionNCE_encoder(nn.Module):
       def __init__(
              self,
              name: str="DecisionNCE-T",
              device: int=0,
              *args,
              **kwargs
       ):
             super().__init__()
             self.name = name
             self.device = device
             
             self.model = DecisionNCE.load(name, device=device)
             self.v_dim = self.model.model.visual.output_dim 
             self.l_dim = self.v_dim
             
       def encode_image(self, imgs: torch.Tensor):
              image_features = self.model.encode_image(imgs)
              return image_features
       
       @torch.no_grad()
       def embed_text(self, langs: list):
              lang_features = self.model.encode_text(langs)
              return lang_features


class DecisionNCE_visual(nn.Module):
       def __init__(
              self,
              mm_encoder: DecisionNCE_encoder,
       ):
              super().__init__()
              self.mm_encoder = mm_encoder
       
       def forward(self, imgs: torch.Tensor):
              image_features = self.mm_encoder.encode_image(imgs)
              return image_features
              
              

class DecisionNCE_lang():
       def __init__(
              self,
              mm_encoder: DecisionNCE_encoder,
       ):
              super().__init__()
              self.mm_encoder = mm_encoder
       
       @torch.no_grad()
       def embed_text(self, langs: list):
              lang_features = self.mm_encoder.embed_text(langs)
              return lang_features


class DecisionNCE_visual_diff():
	def __init__(self, mm_encoder: DecisionNCE_encoder):
		super().__init__()
		self.mm_encoder = mm_encoder
		self.mm_encoder.eval()

	def embed_frame(self, img_begin, img_end):
		image_feature = self.mm_encoder.encode_image(img_begin)
		image_feature_end = self.mm_encoder.encode_image(img_end)
		image_features_diff = image_feature_end - image_feature
		return image_features_diff

	@torch.no_grad()
	def embed_text(self, langs: list):
		lang_features = self.mm_encoder.embed_text(langs)
		return lang_features

              