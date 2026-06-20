import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchreid

# Standard ReID image preprocessing
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def build_reid_model(name='osnet_x1_0', weights_path=None, device='cpu', pretrained=True):
    """
    Builds the TorchReID model and loads weights if a path is provided.
    """
    # If a weights_path is provided, we build without PyPI pretraining first,
    # then load custom weights.
    is_pretrained = pretrained if weights_path is None else False
    
    model = torchreid.models.build_model(
        name=name,
        num_classes=0, # num_classes=0 for feature extraction
        pretrained=is_pretrained
    )
    
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        # Filter out classifier keys since num_classes=0 (feature extraction)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}
        model.load_state_dict(filtered_state_dict, strict=False)
        
    model.to(device)
    model.eval()
    return model

def crop_and_preprocess(frame, bbox, device='cpu'):
    """
    Crops the frame at bbox coordinates and applies preprocessing.
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return preprocess(crop).unsqueeze(0).to(device)

@torch.no_grad()
def extract_features(crops, model):
    """
    Extracts L2-normalized embeddings for a list of preprocessed crops.
    """
    if len(crops) == 0:
        return None
    batch = torch.cat(crops, dim=0)
    feats = model(batch)
    feats = F.normalize(feats, dim=1)
    return feats.cpu().numpy()

def aggregate_embeddings(embs):
    """
    Aggregates list of embeddings for a track by taking the mean and normalizing.
    """
    if embs is None or len(embs) == 0:
        return None
    mean_emb = np.mean(embs, axis=0)
    return mean_emb / (np.linalg.norm(mean_emb) + 1e-6)

def match_embedding(query_emb, gallery, threshold=0.7):
    """
    Computes cosine similarity against stored identities in gallery.
    Returns matched gallery entry and similarity score if above threshold.
    """
    if len(gallery) == 0:
        return None, 0.0
    gallery_feats = np.stack([entry['embedding'] for entry in gallery])
    sims = np.dot(query_emb, gallery_feats.T)
    max_idx = np.argmax(sims)
    if sims[max_idx] >= threshold:
        return gallery[max_idx], sims[max_idx]
    return None, sims[max_idx]
