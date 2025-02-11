import clip
import torch


class ClipEncoder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("RN50", device=self.device)

    @torch.no_grad()
    def embed_text(self, lang):
        text_token = clip.tokenize(lang).to(self.device)
        text_features = self.model.encode_text(text_token)
        return text_features