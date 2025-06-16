import torch
from models import GazePredictorHandler as GazePredictor
from utils import config as cfg, update_config

# Load your config file
update_config("demo/configs/infer_res18_x128_all_vfhq_vert.yaml")

# Initialize full predictor
predictor = GazePredictor(cfg.PREDICTOR, device=cfg.DEVICE)

# Access internal PyTorch model
model = predictor.model
model.eval()

# Correct dummy input: 9 channels
dummy_input = torch.randn(1, 9, 128, 128).to(cfg.DEVICE)

# Export to ONNX
torch.onnx.export(model, dummy_input, "gaze_estimation.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=11)

print("✅ Successfully exported to gaze_estimation.onnx")
