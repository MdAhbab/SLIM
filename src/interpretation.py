import torch

class Interpreter:
    def __init__(self, model):
        self.model = model
        self.grads = None
        self.acts = None

    def get_cam(self, seq, epi, enh_idx=None, prom_idx=None):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Default indices to midpoint if not provided
        if enh_idx is None:
            enh_idx = torch.tensor([2500.0], device=device)
        if prom_idx is None:
            prom_idx = torch.tensor([2500.0], device=device)

        # Hook into Sequence CNN
        target_layer = self.model.seq_cnn[4] # The second Conv1d
        
        def fwd_hook(m, i, o): self.acts = o
        def bwd_hook(m, gi, go): self.grads = go[0]
        
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        
        # Inference
        out, _, _ = self.model(
            seq.unsqueeze(0), epi.unsqueeze(0),
            enh_idx.unsqueeze(0), prom_idx.unsqueeze(0))
        
        # Backward
        self.model.zero_grad()
        out.backward()
        
        # CAM Calculation
        weights = torch.mean(self.grads, dim=2)[0] # Global pool over length
        cam = torch.zeros(self.acts.shape[2]).to(self.acts.device)
        
        for i, w in enumerate(weights):
            cam += w * self.acts[0, i, :]
            
        cam = torch.relu(cam) # ReLU
        
        h1.remove()
        h2.remove()
        return cam.detach().cpu().numpy()