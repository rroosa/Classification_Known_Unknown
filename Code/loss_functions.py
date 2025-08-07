import torch
import torch.nn.functional as F
import torch.nn as nn

def jensen_shannon_divergence(p, q):
    """
     Jensen-Shannon Divergence tra due distribuzioni di probabilità  p e q.
    
        p (torch.Tensor): Distribuzione target (batch_size, num_classes)
        q (torch.Tensor): Distribuzione predetta (batch_size, num_classes)

    restituisce:
        torch.Tensor: Valore della JSD per il batch
    """
    p = p + 1e-8  # Evita problemi di log(0)
    q = q + 1e-8

    m = 0.5 * (p + q)  # Distribuzione media

    kl_p_m = F.kl_div(m.log(), p, reduction='batchmean')  # KL(P || M)
    kl_q_m = F.kl_div(m.log(), q, reduction='batchmean')  # KL(Q || M)

    return 0.5 * (kl_p_m + kl_q_m)

class JSD_Loss(torch.nn.Module):
    def __init__(self):
        super(JSD_Loss, self).__init__()

    def forward(self, p, q):
        return jensen_shannon_divergence(p, q)

