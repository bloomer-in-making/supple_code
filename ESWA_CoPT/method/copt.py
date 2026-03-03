import torch
import torch.nn as nn
import torch.nn.functional as F

class CoPT(nn.Module):
    """ CoPT (Context-based Prompt Tuning) """
    def __init__(self, cfg, prefix_len=10, emb_size=768): 
        super(CoPT, self).__init__()

        # hyp
        self.hidden_size = cfg.DPT.HIDDEN_SIZE
        
        self.prefix_len = prefix_len 
        self.emb_size = emb_size

        
        # prompt 
        self.prompt_first = nn.Parameter(torch.empty(self.prefix_len, self.hidden_size))
        self.prompt_second = nn.Parameter(torch.empty(self.hidden_size - 1, self.emb_size))

    def init_prompt(self):
        random_range = 0.5 
        self.prompt_first.data.uniform_(-random_range, random_range)
        self.prompt_second.data.uniform_(-random_range, random_range)

    
    def generate(self, emb_x = None):
        dist_token = torch.mean(emb_x, dim=1) 
        dist_token = torch.mean(dist_token, dim=0, keepdim=True)  # 1,512
        
        x1 = self.prompt_first
        x2 = torch.concat([self.prompt_second, dist_token], dim=0) #[9,512] + [1,512] = [10,512]
            
        prompt_token = torch.matmul(x1, x2) 
        return prompt_token
