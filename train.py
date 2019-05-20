import argparse
import torch
from torch.utils import data as data
from torch.autograd import Variable
from datasets import dataset
from model import Model
from vis_tool import Visualizer
import numpy as np
import os
from tqdm import *

# parser
class train_options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='training options of CADN')

        # data related
        self.parser.add_argument('--data_roots', type=str, default='OCTdata')
        self.parser.add_argument('--nThreads', type=int, default=4)
        self.parser.add_argument('--img_channels', type=int, default=1)
        self.parser.add_argument('--scale_factor', type=int, default=2)

        # network related
        self.parser.add_argument('--num_features', type=int, default=128)
        self.parser.add_argument('--num_blocks', type=int, default=3)

        # training related
        self.parser.add_argument('--num_epochs', type=int, default=1000)
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--train_interval', type=int, default=10)
        self.parser.add_argument('--num_critics', type=int, default=3)
        self.parser.add_argument('--step', type=int, default=100)

        # resume train related
        self.parser.add_argument('--resume', type=bool, default=False)
        self.parser.add_argument('--start_epoch', type=int, default=1)
        self.parser.add_argument('--G_AB_checkpoint', type=str, default='')
        self.parser.add_argument('--G_BA_checkpoint', type=str, default='')
        self.parser.add_argument('--D_A_checkpoint', type=str, default='')
        self.parser.add_argument('--D_B_checkpoint', type=str, default='')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt


# train
class trainer():
    def __init__(self, ):
        super(trainer, self).__init__()
        # parse option
        self.train_parser = train_options()
        self.opts = self.train_parser.parse()
        # train data
        print('loading training data...')
        self.datasets = dataset(self.opts.data_roots, self.opts.scale_factor)
        self.train_dataloader = data.DataLoader(self.datasets, self.opts.batch_size, shuffle=True,
                                                num_workers=self.opts.nThreads)
        # device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # model
        self.model = Model(self.opts, self.device)
        # visualizer
        self.train_vis = Visualizer(env='Training')

    # adjustable learning rate
    def adjust_learning_rate(self, epoch):
        lr = self.opts.lr * (0.1 ** (epoch // self.opts.step))
        if lr < 1e-7:
            lr = 1e-7
        return lr

    def train_process(self, model, start_epoch):
        for epoch in range(start_epoch, self.opts.num_epochs):
            lr = self.adjust_learning_rate(epoch - 1)
            for param_group in self.model.G_AB_optim.param_groups:
                param_group["lr"] = lr
                print("epoch =", epoch, "lr =", self.model.G_AB_optim.param_groups[0]["lr"])
            for param_group in self.model.G_BA_optim.param_groups:
                param_group["lr"] = lr
                print("epoch =", epoch, "lr =", self.model.G_BA_optim.param_groups[0]["lr"])
            for param_group in self.model.D_A_optim.param_groups:
                param_group["lr"] = lr
                print("epoch =", epoch, "lr =", self.model.D_A_optim.param_groups[0]["lr"])
            for param_group in self.model.D_B_optim.param_groups:
                param_group["lr"] = lr
                print("epoch =", epoch, "lr =", self.model.D_B_optim.param_groups[0]["lr"])

            for i, (imageX, imageY, imageZ) in enumerate(self.train_dataloader):
                imageX, imageY, imageZ = Variable(imageX), Variable(imageY), Variable(imageZ)
                imageX, imageY, imageZ = imageX.to(self.device), imageY.to(self.device), imageZ.to(self.device)
                # print(imageZ)

                model.G_AB_optim.zero_grad()
                model.G_BA_optim.zero_grad()
                model.D_A_optim.zero_grad()
                model.D_B_optim.zero_grad()


                # -----------------------------------
                # ------traing discriminator---------
                # -----------------------------------

                # clearX: the clear vision of noisy imageX
                # noisyY: the noisy vision of clear imageY
                clearX = model.G_AB(imageX)
                noisyY = model.G_BA(imageY)

                # adversarial loss
                dis_noisy_X = model.D_A(imageX)
                dis_noisy_Y = model.D_A(noisyY)
                real = Variable(torch.ones(dis_noisy_Y.size())).to(self.device)
                fake = Variable(torch.zeros(dis_noisy_X.size())).to(self.device)
                L_dis_noisy = 0.5 * model.criterionL2(dis_noisy_X, real) + 0.5 * model.criterionL2(dis_noisy_Y, fake) + model.criterionL2(dis_noisy_Y, real)


                dis_clear_X = model.D_B(clearX)
                dis_clear_Y = model.D_B(imageY)
                real = Variable(torch.ones(dis_clear_Y.size())).to(self.device)
                fake = Variable(torch.zeros(dis_clear_X.size())).to(self.device)
                L_dis_clear = 0.5 * model.criterionL2(dis_clear_Y, real) + 0.5 * model.criterionL2(dis_clear_X, fake) + model.criterionL2(dis_clear_X, real)

                L_adv = L_dis_noisy + L_dis_clear
                d_loss = L_adv
                d_loss.backward()
                model.D_A_optim.step()
                model.D_B_optim.step()

                if i % self.opts.num_critics == 0:
                    # clearX: the clear vision of noisy imageX
                    # noisyY: the noisy vision of clear imageY
                    clearX = model.G_AB(imageX)
                    noisyY = model.G_BA(imageY)

                    # adversarial loss
                    dis_noisy_X = model.D_A(imageX)
                    dis_noisy_Y = model.D_A(noisyY)
                    real = Variable(torch.ones(dis_noisy_Y.size())).to(self.device)
                    fake = Variable(torch.zeros(dis_noisy_X.size())).to(self.device)
                    L_dis_noisy = 0.5 * model.criterionL2(dis_noisy_X, real) + 0.5 * model.criterionL2(dis_noisy_Y,
                                                                                                       fake) + model.criterionL2(
                        dis_noisy_Y, real)

                    dis_clear_X = model.D_B(clearX)
                    dis_clear_Y = model.D_B(imageY)
                    real = Variable(torch.ones(dis_clear_Y.size())).to(self.device)
                    fake = Variable(torch.zeros(dis_clear_X.size())).to(self.device)
                    L_dis_clear = 0.5 * model.criterionL2(dis_clear_Y, real) + 0.5 * model.criterionL2(dis_clear_X,
                                                                                                       fake) + model.criterionL2(
                        dis_clear_X, real)

                    L_adv = L_dis_noisy + L_dis_clear

                    # Cyclic loss
                    noisyX = model.G_BA(clearX)
                    clearY = model.G_AB(noisyY)

                    L_cyclic = model.criterion_cyclic(imageX, noisyX) + model.criterion_cyclic(imageY, clearY)

                    # identity loss
                    L_identity = model.criterion_identity(model.G_AB(imageY), imageY) + \
                                 model.criterion_identity(model.G_BA(imageX), imageX)

                    g_loss = L_adv + 10 * L_cyclic + 5 * L_identity
                    g_loss.backward()

                    model.G_AB_optim.step()
                    model.G_BA_optim.step()
                    model.D_A_optim.step()
                    model.D_B_optim.step()

                    # vis
                    idx = np.random.choice(self.opts.batch_size)
                    images_noise_row = {'imageX': imageX[idx].detach().cpu().numpy(),
                                        'clearX': clearX[idx].clamp(0, 1).mul(255).detach().cpu().numpy(),
                                        'noisyX': noisyX[idx].clamp(0, 1).mul(255).detach().cpu().numpy()}
                    images_clear_row = {'imageY': imageY[idx].detach().cpu().numpy(),
                                        'noisyY': noisyY[idx].clamp(0, 1).mul(255).detach().cpu().numpy(),
                                        'clearY': clearY[idx].clamp(0, 1).mul(255).detach().cpu().numpy()}
                    losses = {'g_loss': g_loss.item(), 'adversarial loss': L_adv.item(),
                              'cyclic loss': L_cyclic.item(), 'identity loss': L_identity.item()}
                    vis_images(self.train_vis, images_noise_row)
                    vis_images(self.train_vis, images_clear_row)
                    vis_loss(self.train_vis, losses)

                    print('[{}/{}] [{}/{}] loss:{}'.format(epoch, self.opts.num_epochs, i, len(self.train_dataloader), g_loss.item()))

            if epoch % self.opts.train_interval == 0:
                models = {'G_AB': model.G_AB,
                          'G_BA': model.G_BA,
                          'D_A': model.D_A,
                          'D_B': model.D_B}
                save = save_model(models, epoch)
                save.save_checkpoint()

    # first training
    def first_train(self):
        self.train_process(self.model, self.opts.start_epoch)

    # resume training
    def resume_train(self):
        # load model parameters
        G_AB_checkpoint = torch.load(self.opts.G_AB_checkpoint)
        self.model.G_BA.load_state_dict(G_AB_checkpoint['model'].state_dict())
        G_BA_checkpoint = torch.load(self.opts.G_BA_checkpoint)
        self.model.G_AB.load_state_dict(G_BA_checkpoint['model'].state_dict())
        D_A_checkpoint = torch.load(self.opts.D_A_checkpoint)
        self.model.D_A.load_state_dict(D_A_checkpoint['model'].state_dict())
        D_B_checkpoint = torch.load(self.opts.D_B_checkpoint)
        self.model.D_B.load_state_dict(D_B_checkpoint['model'].state_dict())
        # train
        self.train_process(self.model, self.opts.start_epoch)

    def train(self):
        if self.opts.resume:
            print('resume training at epoch {}...'.format(self.opts.start_epoch))
            self.resume_train()
        else:
            print('start first training...')
            self.first_train()


# visualizer
def vis_images(vis, images):
    vis.img_many(images)


def vis_loss(vis, losses):
    vis.plot_many(losses)


class save_model():
    def __init__(self, models, epoch):
        self.model_folder = "model_para/"
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        self.models = models
        self.epoch = epoch

    def save_checkpoint(self):
        for (key, value) in self.models.items():
            checkpoint_path = self.model_folder + '{}_{}.pkl'.format(key, self.epoch)
            state_dict = {'epoch': self.epoch, 'model': value}
            torch.save(state_dict, checkpoint_path)
            print("Checkpoint saved to {}".format(checkpoint_path))


if __name__ == '__main__':
    train = trainer()
    train.train()
















