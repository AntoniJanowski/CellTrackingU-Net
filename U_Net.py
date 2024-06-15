import torch
import torch.nn as nn
from U_Net_helper_functions import *
import lightning.pytorch as pl

class Convolutional_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride = 1, dropout = 0.1):
        super(Convolutional_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout(p=dropout, inplace=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        residual = x
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = x + residual
        return x
    

class UNet(pl.LightningModule):
    def __init__(self, list_of_chanel_numbers, crop = None, loss_fn = torch.nn.CrossEntropyLoss(), lr = 0.001, betas = (0.9, 0.999), kernel_size=3, dropout = 0.1, padding = 1,
                  log = False):
        super(UNet, self).__init__()
        self.list_of_chanel_numbers = list_of_chanel_numbers
        self.channel_pairs = [(list_of_chanel_numbers[i], list_of_chanel_numbers[i+1]) for i in range(len(list_of_chanel_numbers) - 1)]
        self.bottom_path = nn.ModuleList()
        self.top_path = nn.ModuleList()
        self.UpConvs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2)
        for pair in self.channel_pairs:
            c_in, c_out = pair
            self.bottom_path.append(Convolutional_Block(c_in, c_out, kernel_size=kernel_size, dropout = dropout, padding = padding))
        self.bottom_block = Convolutional_Block(list_of_chanel_numbers[-1], list_of_chanel_numbers[-1], kernel_size=kernel_size, dropout = dropout, padding=padding)
        self.channel_pairs[0] = (2, list_of_chanel_numbers[1]) # I want the model to return a tensor of shape 2 x H x W, because we want to predict only two classes
        self.channel_pairs.reverse()
        for pair in self.channel_pairs:
            c_out, c_in = pair
            self.UpConvs.append(torch.nn.ConvTranspose2d(c_in, c_in, kernel_size = 2, stride = 2, ))
            self.top_path.append(Convolutional_Block(c_in, c_out, kernel_size=kernel_size, dropout = dropout, padding=padding))

        self.crop = crop #for testing purposes, later this will be implemented into the dataloader
        self.loss_fn = loss_fn
        self.lr = lr
        self.betas = betas
        self.log = log
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout

    def forward(self, x):
        residuals = []
        for i, module in enumerate(self.bottom_path):
            # print(f'starting to process module {i}, x.shape = {x.shape}')
            x = module(x)
            residuals.append(x)
            if i < len((self.bottom_path)):
                # print(f'POOOOOOLING {i}')
                x = self.pool(x)
            # residuals.append(x)
            # print(f'Finished processing module {i}, x.shape = {x.shape}')

        residuals.reverse()
        x = self.bottom_block(x)
        # print(f'bottom path over, x.shape: {x.shape}')

        for i, module in enumerate(self.top_path):
            # print(f'starting to process module {i}, x.shape = {x.shape}')
            if i >= 0:
                # print(f'UP CONV: x.shape before: {x.shape}')
                x = self.UpConvs[i](x)
                # print(f'UP CONV: x.shape after : {x.shape}')
            x = x + residuals[i]
            x = module(x)
            # print(f'Finished processing module {i}, x.shape = {x.shape}')
            
        return x
    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        if self.crop != None:
            inputs = inputs[:, :, :self.crop, :self.crop]
            labels = labels[:, :, :self.crop, :self.crop]
        
        labels = labels.squeeze().long()
        outputs = self.forward(inputs)
        loss = self.loss_fn(outputs, labels)

        print(f'   TRRAINING: Batch {batch_idx}, loss {loss}')
        if self.log == True:
            self.log('Training Loss', loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            inputs, labels = batch

            if self.crop != None:
                inputs = inputs[:, :, :self.crop, :self.crop]
                labels = labels[:, :, :self.crop, :self.crop]
            
            labels = labels.squeeze().long()
            outputs = self.forward(inputs)
            loss = self.loss_fn(outputs, labels)

            print(f'   VALIDATION: Batch {batch_idx}, loss {loss}')
            if self.log == True:
                self.log('Training Loss', loss, on_step=True, on_epoch=True)
            return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr, betas = self.betas)
    
    def generate_numpy_output_from_single_image(self, img):
        '''
        Args: img: a tensor of shape (1, 2, 512, 512). It is suposed to be the output of the model. This function will turn it 
        into a numpy array of shape (512, 512) ready to be plotted.
        '''
        model_return_numpy = channel_comparison(img)
        return model_return_numpy
    
    def run_model_on_validation_dataloder(self, val_dataloader):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            batches = 0
            for data in val_dataloader:
                batches +=1
                inputs, labels = data
                labels = labels.squeeze().long()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                
                if batches > 100:
                    break
            print('Avarage validation loss per batch: ', total_loss / batches)
            return total_loss / batches
    
    # def print_model_outputs(self, output, og = None, label = None):
