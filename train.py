import os
import sys
import pyrootutils
import torch
from torch import optim
from tqdm import tqdm
from distutils.util import strtobool

from src.model.deephic import Generator as DeepHiC_Generator  # Import DeepHiC Generator model
from src.model.hicplus import Net as HiCPlus_Generator
from src.model.HiCAlign import SimpleHiCAlign as SimpleHiCAlign
from src.model.hicedrn_Diff import hicedrn_Diff  # baseline models' modules
from src.hicdiff_condition import GaussianDiffusion as Gaussiandiff_cond  # baseline models' modules conditional Diff
from src.hicdiff import GaussianDiffusion as Gaussiandiff # baseline models's modules without conditional
import wandb  # the logger
import argparse

from processdata.PrepareData_linear import GSE130711Module as GSE130711_cond # the datasets
from processdata.PrepareData_linear import GSE131811Module as GSE131811_cond # the datasets
from processdata.PrepareData_linear_sing import GSE130711Module as GSE130711_s # the datasets
from processdata.PrepareData_linear_sing import GSE131811Module as GSE131811_s # the datasets



root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

def create_parser():
    parser = argparse.ArgumentParser(description = 'HiCDiff works for single-cell HI-C data denoising !!!')
    # parser.add_argument('-u', '--unspervised', type = bool, default = True, help = 'True means you will use unsupervsed way to train your model, False indicates you will use supervised way to train your model')
    parser.add_argument('-u', '--unsupervised', action=argparse.BooleanOptionalAction,
                        help="Use 'True' for unsupervised training, 'False' for supervised training")
    parser.add_argument('-b', '--batch_size', type = int, default = 64, help = 'Batch size for embeddings generation.')
    parser.add_argument('-e', '--epoch', type = int, default = 400, help = 'Number of epochs used for embeddings generation')
    parser.add_argument('-l', '--celline', type = str, default = "Human",
                        help = "Which cell line you want to choose for your dataset, default is 'Human', you should choose one name in ['Human', 'Dros']")
    parser.add_argument('-n', '--celln', type = int, default = 1,
                        help = "Cell number in the dataset you want to feed in you model")

    parser.add_argument('-s', '--sigma', type = float, default = 1,
                        help = f"The Gaussian noise level for the raw dataset, it should be equal or larger than 0.0 but not larger than 1.0, '1.0' means the largest noise added to datasets.")

    args = parser.parse_args()
    # args.unsupervised = args.unsupervised.lower() in ["true", "1", "t", "yes"]

    return args

class HiCDiff:
    def __init__(self, epoch = 500, timestep = 1000, cell_Line = 'Human',  cellNo = 1, res = 40000, batch_size = 64, piece_s = 64, sigma = 0.1, condition = True, deg='deno'):
        # initialize the parameters that will be used during fit model
        self.epoch = epoch
        self.cell_Line = cell_Line
        self.cellNo = cellNo
        self.res = res
        self.chunk = piece_s
        self.sigma = sigma
        self.deg = deg
        self.condition = condition


        # whether using GPU for training
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.device = device

        # out_dir: directory storing checkpoint files and parameters for saving to the our_dir
        dir_name = 'Model_Weights'
        self.out_dir = os.path.join(root, dir_name)
        #self.out_dirM = os.path.join(root, "Metrics")
        os.makedirs(self.out_dir, exist_ok=True)  # makedirs will make all the directories on the path if not exist.
        #os.makedirs(self.out_dirM, exist_ok=True)

        # prepare training and valid dataset
        if self.cell_Line == 'Human':
            DataModule = GSE130711_s(batch_size=batch_size, res=res, cell_line=cell_Line, cell_No=cellNo, sigma_0=self.sigma, deg=self.deg)
        else:
            DataModule = GSE131811_s(batch_size = batch_size, res = res, piece_size = piece_s, cell_line = cell_Line, cell_No = cellNo)

        DataModule.prepare_data()
        DataModule.setup(stage='fit')

        self.train_loader = DataModule.train_dataloader()
        self.valid_loader = DataModule.val_dataloader()

        if not self.condition:  # Supervised mode
            print("Supervised Mode: Using DeepHiC Expert Model!")
            
            # self.expert_model = HiCPlus_Generator(D_in=40, D_out=24).to(self.device)
            self.expert_model = SimpleHiCAlign().to(self.device)
            self.expert_model.eval()  

            model = hicedrn_Diff(self_condition=True)
            self.diffusion = Gaussiandiff_cond(
                model,
                image_size=piece_s,
                timesteps=timestep,
                loss_type='l2',  # L1 or L2
                beta_schedule='linear',
                auto_normalize=False
            ).to(device)

        else:  # Unsupervised mode
            print("Unsupervised Mode: No Expert Model Used!")
            self.expert_model = None

            model = hicedrn_Diff()
            self.diffusion = Gaussiandiff(
                model,
                image_size=piece_s,
                timesteps=timestep,
                loss_type='l2',  # L1 or L2
                beta_schedule='linear',
                auto_normalize=False
            ).to(device)

    def fit_model(self):
        # optimizer
        optimizer = optim.Adam(self.diffusion.parameters(), lr=2e-5)

        best_ssim = 0
        best_loss = 10000
        for epoch in range(1, self.epoch + 1):
            self.diffusion.train()
            run_result = {'nsamples': 0, 'loss': 0}

            train_bar = tqdm(self.train_loader)
            for batch_data in train_bar:  # target is the pure image without

                data, target, _, info = batch_data
                batch_size = data.shape[0]
                run_result['nsamples'] += batch_size
                data = data.to(self.device)
                target = target.to(self.device)
                # if not self.condition:
                #     x = [data, target]
                # else:
                #     x = target
                
                # 🛠 **Use Expert Model in Supervised Mode** 🛠
                if not self.condition:  # If supervised (conditioned)
                    with torch.no_grad():
                        
                        # data = data.repeat(1, 3, 1, 1)
                        # print(data.shape)
                        # print(f"Input to DeepHiC: min={data.min().item()}, max={data.max().item()}")
                        print(f"Input : min={data.min().item()}, max={data.max().item()}, shape={data.shape}")  
                        expert_target = self.expert_model(data)  # Get cleaned target from DeepHiC
                        print(f"Output: min={expert_target.min().item()}, max={expert_target.max().item()}, shape={expert_target.shape}")
                        # print(f"DeepHiC Output: min={expert_target.min().item()}, max={expert_target.max().item()}")
                    x = [data, expert_target]  # Input: noisy + expert guidance
                    
                else:  
                    x = target  # Unsupervised uses just the noisy target

                loss = self.diffusion(x)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                run_result['loss'] +=loss.item() * batch_size
                train_bar.set_description(desc=f"[{epoch}/{self.epoch}] training Loss: {run_result['loss'] / run_result['nsamples']:.6f}")

            train_loss = run_result['loss'] / run_result['nsamples']

            # =================VALIDATION============================
            valid_result = {'nsamples': 0, 'loss': 0}
            self.diffusion.eval()
            valid_bar = tqdm(self.valid_loader)
            batch_id = 0
            with torch.no_grad():
                for batch_data in valid_bar:   # data is the pure image without noise
                    data, target, _, info = batch_data
                    batch_size = data.shape[0]
                    valid_result['nsamples'] += batch_size
                    data = data.to(self.device)
                    target = target.to(self.device)

                    # if not self.condition:
                    #     x = [data, target]
                    # else:
                    #     x = target
                    if not self.condition:
                        # data = data.repeat(1, 3, 1, 1)
                        expert_target = self.expert_model(data)  # Get expert supervision
                        x = [data, expert_target]
                    else:
                        x = target

                    loss = self.diffusion(x)

                    '''
                    #sample_out = self.diffusion.valuate(x)
                    if batch_id == 0:
                        out = self.diffusion.super_resolution(data)
                        print(f'the data shape is {data.shape} predicted results shape is {out.shape}.')
                        out = inverse_data_transform('rescaled', out)
                        hr = inverse_data_transform('rescaled', target)
                        batch_ssim = ssim(out, hr)
                        batch_mse = ((out - hr) ** 2).mean()
                        batch_psnr = 10 * log10(1 / (batch_mse))
                        print(f'the ssim is {batch_ssim} and the psnr is {batch_psnr}\n')
                    batch_id += 1
                    '''

                    valid_result['loss'] += loss.item() * batch_size
                    valid_bar.set_description(
                        desc=f"[{epoch}/{self.epoch}] Validation Loss: {valid_result['loss'] / valid_result['nsamples']:.6f}")

                valid_loss = valid_result['loss'] / valid_result['nsamples']

                # now_ssim = batch_ssim
                now_loss = valid_loss
                if now_loss < best_loss:
                    best_loss = now_loss
                    # print(f'Now, Best ssim is {best_loss:.6f}')
                    condition_str = "cond" if not self.condition else "uncond"
                    best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_{condition_str}_l2_lin.pytorch'
                    torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, best_ckpt_file))
                    # best_ckpt_file = f'bestg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_cond_l2_lin.pytorch'
                    # torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, best_ckpt_file))
                #wandb.log({"Epoch": epoch, 'train/loss':train_loss,'valid/loss': valid_loss})

        # Inside your fit_model() method, in the final checkpoint saving section:
        condition_str = "cond" if not self.condition else "uncond"  # supervised: 'cond'; unsupervised: 'uncond'

        final_ckpt_file = f'finalg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_{condition_str}_l2_lin.pytorch'
        torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, final_ckpt_file))

        # final_ckpt_file = f'finalg_{self.res}_c{self.chunk}_s{self.chunk}_{self.cell_Line}{self.cellNo}_HiCedrn_cond_l2_lin.pytorch'
        # torch.save(self.diffusion.state_dict(), os.path.join(self.out_dir, final_ckpt_file))

if __name__ == "__main__":

    args = create_parser()
    train_model = HiCDiff(epoch = args.epoch, batch_size = args.batch_size, cellNo = args.celln, cell_Line = args.celline, sigma = args.sigma, condition = args.unsupervised)
    # train_model.fit_model()

    print("DeepHiC Model Loaded Successfully!")
    # print(train_model.expert_model)  # Print model architecture to verify it loaded correctly

    train_model.fit_model()
    print("Training is done!!! ~~~~~")
    print("Training is done !!! ~~~~~")
