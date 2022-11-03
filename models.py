from model_utils import *
import os
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
class base_AE(nn.Module):
    def __init__(self):
        super(base_AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(5, 2),
            torch.nn.ReLU()

        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 5)

        )

    def forward(self, x):
        f = self.encoder(x)
        output = self.decoder(f)
        return {'output': output, 'latent': f}


class MemAE(nn.Module):
    def __init__(self, mem_dim=100, shrink_thres=0.0025):
        super(MemAE, self).__init__()


        self.encoder = nn.Sequential(

            nn.Linear(5, 2),
            torch.nn.ReLU()

        )
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=2, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            nn.Linear(2, 5)

        )

    def forward(self, x):
        f = self.encoder(x)
        res_mem = self.mem_rep(f)
        f = res_mem['output']
        att = res_mem['att']
        output = self.decoder(f)
        return {'output': output, 'att': att, 'latent': f}


class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return {'output': x_hat, 'mean': mean, 'var': log_var, 'latent': z}


class MVAE(nn.Module):
    def __init__(self, Encoder, Decoder, mem_dim=100, shrink_thres=0.0025):
        super(MVAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=2, shrink_thres=shrink_thres)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var

        res_mem = self.mem_rep(z)
        f = res_mem['output']
        att = res_mem['att']

        x_hat = self.Decoder(f)

        return {'output': x_hat, 'mean': mean, 'var': log_var, 'att': att, 'latent': f}


def make_dir(model):
    if not os.path.exists(f'./{model}'):
        os.mkdir(f'./{model}')


class model():

    def __init__(self, args, train, train_n, test, test_faulty, norm):

        self.args = args
        self.norm=norm
        self.train = train
        self.train_n = train_n
        self.test = test
        self.test_faulty = test_faulty
        self.model_ = None
        self.loss_function = torch.nn.MSELoss()
        self.tr_entropy_loss_func = EntropyLossEncap()
        self.optimizer =None
        self.entropy_loss_weight = 0.0002
        self.mem = False
        self.entropy_loss_weight = 0

        print(f"{self.args.model} is selected")
        make_dir(self.args.model)

        match self.args.model:
            case "AE":
                self.model_ = base_AE()
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)
            case "DAE":
                self.model_ = base_AE()
                self.train = self.train_n
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "MAE":
                self.model_ = MemAE(mem_dim=self.args.memdim, shrink_thres=0.0025)
                self.mem = True
                self.entropy_loss_weight = 0.0002
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "VAE":
                encoder = Encoder(input_dim=5, hidden_dim=3, latent_dim=2)
                decoder = Decoder(latent_dim=2, hidden_dim=3, output_dim=5)
                self.model_ = VAE(Encoder=encoder, Decoder=decoder)
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

            case "MVAE":
                encoder = Encoder(input_dim=5, hidden_dim=3, latent_dim=2)
                decoder = Decoder(latent_dim=2, hidden_dim=3, output_dim=5)
                self.model_ = MVAE(Encoder=encoder, Decoder=decoder, mem_dim=self.args.memdim, shrink_thres=0.0025)
                self.entropy_loss_weight = 0.0002
                self.mem = True
                self.optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.0001)

    def plot_memory(self, model, epoch):
        if not os.path.exists(f'./{self.args.model}/MemoryElements'):
            os.mkdir(f'./{self.args.model}/MemoryElements')
        plt.scatter(model.mem_rep.melements.detach().numpy()[:, 0], model.mem_rep.melements.detach().numpy()[:, 1])
        # plt.show()
        plt.ylabel("Feature2")
        plt.xlabel("Feature1")
        plt.title("Memory_Elements_MemAE")
        plt.savefig(f'./{self.args.model}/MemoryElements/Epoch_{epoch}.jpg', bbox_inches="tight", pad_inches=0.0)
        plt.clf()

    def train_model(self):
        wait = 0
        epoch_loss = []
        eot=False # end of training
        for epoch in range(self.args.epochs):

            if self.mem:  # if model has memory module plot the elements during training
                self.plot_memory(self.model_, epoch)



            if (eot == True):
                break
            losses = 0


            print(f"epoch: {epoch}")

            iteration = len(self.train) // self.args.batch
            for i in range(iteration):
                # latent_mae=[]
                # outputs = []

                obs = torch.from_numpy(self.train.iloc[i * self.args.batch:(i + 1) * self.args.batch].to_numpy())

                reconstructed = self.model_(obs.float())
                #  print("obs")
                #  print(obs)
                # print("rec")
                # print(reconstructed['output'][0])

                # print(att_w)
                loss = self.loss_function(reconstructed['output'], obs.float())

                if self.mem:
                    att_w = reconstructed['att']
                    entropy_loss = self.tr_entropy_loss_func(att_w)
                    loss = loss + self.entropy_loss_weight * entropy_loss

                loss_val = loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = losses + loss_val

            epoch_loss.append(losses / iteration)
            print(f"epoch_{epoch}_loss:  {losses / iteration}")

            if len(epoch_loss)>2:
                if epoch_loss[epoch] > epoch_loss[epoch - 1]:
                    wait = wait + 1
                    if wait > self.args.patience:
                        print("End of training")
                        eot = True
                        torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')
                        print("early stopping")

                else:
                    wait = 0

            if (epoch == self.args.epochs - 1):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_final.pt')

            if (epoch % 50 == 0):
                torch.save(self.model_.state_dict(), f'./{self.args.model}/{self.args.model}_snap.pt')

        plt.plot(epoch_loss)
        print(epoch_loss)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.title(f"Training loss for {self.args.model}")
        plt.savefig(f'./{self.args.model}/{self.args.model}_loss.png', bbox_inches="tight", pad_inches=0.0)
        plt.clf()

    def reconstruct(self, dataframe, description="Reconstruction"):

        model_para = torch.load(f'./{self.args.model}/{self.args.model}_final.pt')
        self.model_.load_state_dict(model_para)

        result=[] # reconstructed values
        latent=[]
        for i in range(len(dataframe)):
            obs = torch.from_numpy(dataframe.iloc[i + 0:i + 1].to_numpy())
            reconstructed = self.model_(obs.float())
            result.append(reconstructed['output'].detach().numpy()[0])
            #latent.append(reconstructed['latent'].detach().numpy()[0])

        #result= de_normalize_2d(np.array(result),self.norm)

        df_result = pd.DataFrame(result, columns=self.train.columns)
        df_result  = de_normalize_2d(df_result , self.norm)
        dataframe=  de_normalize_2d(dataframe , self.norm)
        for b in self.train.columns:
            plt.plot(df_result[b].iloc[0:700], linestyle='dotted', color='red', label=f'Reconstructed_{self.args.model}',
                     marker='.')

            plt.plot(dataframe[b].iloc[0:700], label='Actual', color='blue', marker='.')
            plt.legend()
            plt.xlabel(f"{b}_train")
            plt.title(description)
            plt.savefig(f'./{self.args.model}/{self.args.model}_{b}_{description}.jpg.png' , bbox_inches="tight", pad_inches=0.0)
            plt.clf()