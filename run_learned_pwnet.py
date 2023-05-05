import gym
import torch 
import torch.nn as nn
import numpy as np      
import pickle
import time
import toml

from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser
from os.path import join
from games.carracing import RacingNet, CarRacing
from pw_modules.learned_pw_network import LearnedPWNet
from pw_modules.list_module import ListModule
from pw_modules.losses import clust_loss, sep_loss, l1_loss
from ppo import PPO
from torch.distributions import Beta
from tqdm import tqdm


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument('--num_iterations', type=int, default=1)
    parser.add_argument('--model_dir', type=str, default='Car Racing/weights/pw_net.pth')
    parser.add_argument('--config_file', type=str, default='CarRacingConfig.toml')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--prototype_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--delay_ms', type=int, default=0)
    parser.add_argument('--num_prototypes', type=int, default=4)
    parser.add_argument('--simulation_epochs', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=1000)

    parser.add_argument('--lambda1', type=float, default=1., help="Strength of L0 penalty")
    parser.add_argument('--lambda2', type=float, default=.5, help="Coefficient of clustering loss")
    parser.add_argument('--lambda3', type=float, default=.008, help="Coefficient of separation loss")

    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--env', type=str, default="car")

    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
    parser.add_argument('--device', type=str)

    parser.add_argument('--tensorboard', action='store_true', help='Start a tensorboard session and write the results of training.  Only applies to training.')

    args = parser.parse_args(args=args)

    if args.cpu:
        args.cuda = False

    args.device = 'cuda' if args.cuda else 'cpu'

    args = vars(args)
    return args


def evaluate_loader(model, loader, loss, args):
    model.eval()
    total_error = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            imgs, labels = data
            imgs, labels = imgs.to(args['device']), labels.to(args['device'])
            logits = model(imgs)
            current_loss = loss(logits, labels)
            total_error += current_loss.item()
            total += len(imgs)
    model.train()
    return total_error / total


def load_config(args):
    with open(args['config_file'], "r") as f:
        config = toml.load(f)
    return config

def proto_loss(model, nn_human_x, criterion):
    model.eval()
    target_x = trans_human_concepts(model, nn_human_x)
    loss = criterion(model.prototypes, target_x) 
    model.train()
    return loss
    

def trans_human_concepts(model, nn_human_x):
    model.eval()
    trans_nn_human_x = list()
    for i, t in enumerate(model.ts):
        trans_nn_human_x.append( t( torch.tensor(nn_human_x[i], dtype=torch.float32).view(1, -1)) )
    model.train()
    return torch.cat(trans_nn_human_x, dim=0)


def main(args=None):
    args = parse_args(args=args)
    if args['mode'] == 'train':
        train(args)
    else:
        evaluate(args)

def train(args):
    #### Start Collecting Data To Form Final Mean and Standard Error Results
    for _ in range(args['num_iterations']):
        with open('Car Racing/data/X_train.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open('Car Racing/data/real_actions.pkl', 'rb') as f:
            real_actions = pickle.load(f)
        with open('Car Racing/data/obs_train.pkl', 'rb') as f:
            states = pickle.load(f)

        X_train = np.array([item for sublist in X_train for item in sublist])
        real_actions = np.array([item for sublist in real_actions for item in sublist])
        states = np.array([item for sublist in states for item in sublist])
        tensor_x = torch.Tensor(X_train)
        tensor_y = torch.tensor(real_actions, dtype=torch.float32)
        train_dataset = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])

        # Human defined Prototypes for interpretable model (these were gotten manually earlier)
        # A clustering analysis could be done to help guide the search, or they can be manually chosen.
        # Lastly, they could also be learned as pwnet* shows in the comparitive tests
        # p_idxs = np.array([10582, 20116, 4616, 2659])
        # nn_human_x = X_train[p_idxs.flatten()]
        # nn_human_actions = real_actions[p_idxs.flatten()]

        #### Training
        model = LearnedPWNet(args).eval()
        model.to(args['device'])
        # model.nn_human_x.data.copy_( torch.tensor(nn_human_x) )

        mse_loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        best_error = float('inf')
        model.train()

        # Freeze Linear Layer
        # model.linear.weight.requires_grad = False

        if args['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=f'runs/{model.env}_' +
                    f'{model.num_prototypes}protos_' +
                    f'{args["lambda1"]}l1_{args["lambda2"]}l2' +
                    time.strftime("%Y%m%d-%H%M%S"))

        global_step = 0
        for epoch in range(args['num_epochs']):

            running_loss = 0
            model.eval()
            train_error = evaluate_loader(model, train_loader, mse_loss, args)
            model.train()

            if train_error < best_error:
                torch.save(  model.state_dict(), args['model_dir'])
                best_error = train_error

            for instances, labels in train_loader:
                global_step += 1

                optimizer.zero_grad()

                instances, labels = instances.to(args['device']), labels.to(args['device'])

                logits = model(instances)
                loss = mse_loss(logits, labels) + \
                        -(1 / np.prod(model.linear.weights.shape)) * model.linear.regularization() + \
                        args['lambda2'] * clust_loss(instances, labels, model, mse_loss)
                        # args['lambda1'] * l1_loss(model) + \
                        # args['lambda3'] * sep_loss(instances, labels, model, mse_loss)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if args['tensorboard'] and global_step % args['log_every'] == 0:
                    writer.add_scalar('Loss/running error', running_loss / (global_step % len(train_loader)), global_step)
                    writer.add_scalar('Loss/MAE', train_error, global_step)
                    writer.add_figure('prototypes',
                                      model.plot_prototypes(X_train, states),
                                      global_step)
                    writer.add_figure('weights',
                                      model.plot_weights(),
                                      global_step)
                    writer.add_scalars('Proto/prototype dists',
                            dict(zip([str(i) for i in range(model.num_prototypes)], model.prototype_dists)),
                            global_step)

            print("Epoch:", epoch)
            print("Running Error:", running_loss / len(train_loader))
            print("MAE:", train_error)
            print(" ")
            scheduler.step()

        # states, actions, rewards, log_probs, values, dones, X_train = [], [], [], [], [], [], []
        # self_state = ppo._to_tensor(env.reset())


def evaluate(args):
    data_rewards = list()
    data_errors = list()
    with open('Car Racing/data/X_train.pkl', 'rb') as f:
        X_train = pickle.load(f)
    with open('Car Racing/data/real_actions.pkl', 'rb') as f:
        real_actions = pickle.load(f)
    with open('Car Racing/data/obs_train.pkl', 'rb') as f:
        states = pickle.load(f)

    X_train = np.array([item for sublist in X_train for item in sublist])
    real_actions = np.array([item for sublist in real_actions for item in sublist])
    states = np.array([item for sublist in states for item in sublist])
    tensor_x = torch.Tensor(X_train)
    tensor_y = torch.tensor(real_actions, dtype=torch.float32)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args['batch_size'])

    mse_loss = nn.MSELoss()
    for _ in range(args['num_iterations']):
        cfg = load_config(args)
        env = CarRacing(frame_skip=0, frame_stack=4,)
        net = RacingNet(env.observation_space.shape, env.action_space.shape)
        ppo = PPO(
            env,
            net,
            lr=cfg["lr"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            gae_lambda=cfg["gae_lambda"],
            clip=cfg["clip"],
            value_coef=cfg["value_coef"],
            entropy_coef=cfg["entropy_coef"],
            epochs_per_step=cfg["epochs_per_step"],
            num_steps=cfg["num_steps"],
            horizon=cfg["horizon"],
            save_dir=cfg["save_dir"],
            save_interval=cfg["save_interval"],
        )
        ppo.load("Car Racing/weights/agent_weights.pt")

        # Wrapper model with learned weights
        model = LearnedPWNet(args).eval()
        model.load_state_dict(torch.load(args['model_dir']))
        model.to(args['device'])
        print("Sanity Check: MSE Eval:", evaluate_loader(model, train_loader, mse_loss, args))

        reward_arr = []
        all_errors = list()

        for i in tqdm(range(args['simulation_epochs'])):
            state = ppo._to_tensor(env.reset())
            count = 0
            rew = 0
            model.eval()

            for t in range(10000):
                # Get black box action
                value, alpha, beta, latent_x = ppo.net(state)
                value, alpha, beta = value.squeeze(0), alpha.squeeze(0), beta.squeeze(0)
                policy = Beta(alpha, beta)
                input_action = policy.mean.detach()
                bb_action = ppo.env.preprocess(input_action)

                action = model(latent_x)

                all_errors.append(mse_loss( torch.tensor(bb_action).to(args['device']), action[0]).detach().item())

                state, reward, done, _, _ = ppo.env.step(action[0].detach().cpu().numpy(), real_action=True)
                state = ppo._to_tensor(state)
                rew += reward
                count += 1
                if done:
                    break

            reward_arr.append(rew)

        data_rewards.append(  sum(reward_arr) / args['simulation_epochs'])
        data_errors.append(  sum(all_errors) / args['simulation_epochs'])
        print("Iteration Reward:", sum(reward_arr) / args['simulation_epochs'])

    data_errors = np.array(data_errors)
    data_rewards = np.array(data_rewards)

    print(" ")
    print("===== Data MAE:")
    print("Mean:", data_errors.mean())
    print("Standard Error:", data_errors.std() / np.sqrt(args['num_iterations'])  )
    print(" ")
    print("===== Data Reward:")
    print("Rewards:", data_rewards)
    print("Mean:", data_rewards.mean())
    print("Standard Error:", data_rewards.std() / np.sqrt(args['num_iterations'])  )

if __name__=="__main__":
    main()
