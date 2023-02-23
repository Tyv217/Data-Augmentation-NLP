import random, torch, numpy
import pytorch_lightning.utilities.seed as plseed

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.cuda.manual_seed(seed)           
    torch.cuda.manual_seed_all(seed)
    plseed.seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True