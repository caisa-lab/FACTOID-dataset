import torch 

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)
SEED = 1234
