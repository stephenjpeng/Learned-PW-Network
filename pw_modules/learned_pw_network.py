import time
import torch 
import torch.nn as nn
import numpy as np      
import pickle


from pw_modules.list_module import ListModule
from l0_modules.l0_layers import L0Dense
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm


class LearnedPWNet(nn.Module):
    """ Modified from PW-Net code """

    def __init__(self, args):
        super(LearnedPWNet, self).__init__()
        self.ts = ListModule(self, 'ts_')
        self.num_classes = args['num_classes']
        self.num_prototypes = args['num_prototypes']
        self.latent_size = args['latent_size']
        self.prototype_size = args['prototype_size']
        self.lambda1 = args['lambda1']
        self.device = args['device']
        self.env = args['env']

        self.prototype_idxs = [None] * self.num_prototypes
        self.prototype_dists = [None] * self.num_prototypes

        for i in range(self.num_prototypes):
            transformation = nn.Sequential(
                nn.Linear(self.latent_size, self.prototype_size),
                nn.InstanceNorm1d(self.prototype_size),
                nn.ReLU(),
                nn.Linear(self.prototype_size, self.prototype_size),
            )
            self.ts.append(transformation)  

        self.latent_protos = nn.Parameter(
            torch.randn((self.num_prototypes, self.prototype_size), dtype=torch.float32),
            requires_grad=True
        )
        self.aligned_prototypes = []
         
        self.epsilon = 1e-5
        # self.linear = nn.Linear(self.num_prototypes, self.num_classes, bias=False) 
        self.linear = L0Dense(self.num_prototypes, self.num_classes, bias=False, lamba=self.lambda1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU() 
        
    # not used for now
    def __make_linear_weights(self):
        """
        Must be manually defined to connect prototypes to human-friendly concepts
        For example, -1 here corresponds to steering left, whilst the 1 below it to steering right
        Together, they can encapsulate the overall concept of steering
        More could be connected, but we just use 2 here for simplicity.
        """

        custom_weight_matrix = torch.tensor([
                                             [-1., 0., 0.], 
                                             [ 1., 0., 0.],
                                             [ 0., 1., 0.], 
                                             [ 0., 0., 1.],
        ])
        self.linear.weight.data.copy_(custom_weight_matrix.T)   
        
    def __proto_layer_l2(self, x, p):
        b_size = x.shape[0]
        p = p.view(1, self.prototype_size).tile(b_size, 1).to(self.device) 
        c = x.view(b_size, self.prototype_size).to(self.device)      
        l2s = ( (c - p)**2 ).sum(axis=1).to(self.device) 
        act = torch.log( (l2s + 1. ) / (l2s + self.epsilon) ).to(self.device)  
        return act
    
    def __output_act_func(self, p_acts):    
        """
        Use appropriate activation functions for the problem at hand
        Here, tanh and relu make the most sense as they bin the possible output
        ranges to be what the car is capable of doing.
        """

        p_acts.T[0] = self.tanh(p_acts.T[0])  # steering between -1 -> +1
        p_acts.T[1] = self.relu(p_acts.T[1])  # acc > 0
        p_acts.T[2] = self.relu(p_acts.T[2])  # brake > 0
        return p_acts


    def transform(self, x, idx):
        """
        Transform from black box latent space to prototype[idx] space
        """
        return self.ts[idx](x) 
    

    def extract_prototypes(self, X_train):
        print("Starting prototype extraction...")
        start = time.time()

        trans_x = list()
        for i in tqdm(range(len(X_train))):
            img = X_train[i]
            trans_img = list()
            for j, t in enumerate(self.ts):
                with torch.no_grad():
                    x = self.transform(torch.tensor(img, dtype=torch.float32).view(1, -1), j)
                trans_img.append(x[0].tolist())
            trans_x.append(trans_img)
        trans_x = np.array(trans_x)

        nn_xs = []
        dists = []
        for i in range(self.num_prototypes):
            trained_prototype = self.latent_protos.clone().detach()[i].view(1,-1)
            knn = KNeighborsRegressor(algorithm='brute')
            knn.fit(trans_x[:, i, :], list(range(len(trans_x))))
            dist, nn_idx = knn.kneighbors(X=trained_prototype, n_neighbors=1, return_distance=True)
            # print(dist.item(), nn_idx.item())
            nn_xs.append(nn_idx.item())
            dists.append(dist.item())
        print(f"Prototypes extracted in {time.time() - start} seconds")

        self.prototype_idxs = nn_xs
        self.prototype_dists = dists
        return nn_xs


    def plot_prototypes(self, X_train, states):
        prototype_idxs = self.extract_prototypes(X_train)
        ncols = 4
        nrows = 1 + (self.num_prototypes - 1) // ncols
        fig = plt.figure(figsize=(6, 3))
        for i, idx in enumerate(prototype_idxs):
            ax = fig.add_subplot(nrows, ncols, i+1)
            ax.imshow(states[idx])
            ax.set_title(f"Prototype {i+1} (D: {self.prototype_dists[i]:.2f})")
        return fig

    def plot_weights(self):
        t = self.linear.weights.data.clone().detach()
        xlabs = ['', 'steer', 'brake', 'accel']
        ylabs = [''] + [f'p{i+1}' for i in range(self.num_prototypes)]

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(t, cmap='PiYG', vmin=-t.abs().max(), vmax=t.abs().max())
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                ax.annotate(f'{t[i, j].item():.4f}', (j, i), ha="center", va="center")

        ax.set_yticklabels(ylabs)
        ax.set_xticklabels(xlabs)

        return fig

    
    def forward(self, x):
        # Do similarity of inputs to prototypes
        p_acts = list()
        for i, t in enumerate(self.ts):
            action_prototype = self.latent_protos[i]
            p_acts.append( self.__proto_layer_l2( t(x), action_prototype).view(-1, 1) )
        p_acts = torch.cat(p_acts, axis=1)

        # Put though activation function method
        logits = self.linear(p_acts)
        final_outputs = self.__output_act_func(logits)   

        return final_outputs
