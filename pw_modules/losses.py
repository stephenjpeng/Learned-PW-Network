import torch


def clust_loss(x, y, model, criterion):
    """
    Forces each prototype to be close to training data
    """
    
    ps = model.latent_protos  # take prototypes in new feature space
    model = model.eval()
    b_size = x.shape[0]

    for idx, p in enumerate(ps):
        x_trans = model.transform(x, idx)  # transform into new feature space
        target = p.repeat(b_size, 1)
        if idx == 0:
            loss = criterion(x_trans, target) 
        else:
            loss += criterion(x_trans, target)
    model = model.train()  
    return loss


def sep_loss(x, y, model, criterion):
    pass
#     """
#     Force each prototype to be far from each other. Fails when prototypes live in
#     separate spaces
#     """
#     
#     p = model.prototypes  # take prototypes in new feature space
#     model = model.eval()
#     x = model.main(x)  # transform into new feature space
#     loss = torch.cdist(p, p).sum() / ((NUM_PROTOTYPES**2 - NUM_PROTOTYPES) / 2)
#     return -loss 


def l1_loss(model):
    return torch.linalg.vector_norm(model.linear.weight, ord=1)
