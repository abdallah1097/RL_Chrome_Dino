# RL_Chrome_Dino
This repo implements RL Agent to play Chrome Dino Game

# Preparing Conda env

1. Installing Torch: Referring to https://pytorch.org/get-started/locally/ based on your CUDA version:
```
pip3 install torch torchvision torchaudio
```

Double check Torch is installed correctlly:
```
# python3
import torch
torch.cuda.is_available()
```

2. Install Chrome Driver:

```
CHROME_VERSION=$(google-chrome --version | awk '{print $3}' | cut -d. -f1)
wget https://storage.googleapis.com/chrome-for-testing-public/133.0.6943.141/linux64/chromedriver-linux64.zip
unzip chromedriver-linux64.zip 
sudo mv chromedriver-linux64 /usr/local/bin/
sudo chmod +x /usr/local/bin/chromedriver-linux64/
```