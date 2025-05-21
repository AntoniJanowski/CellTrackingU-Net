import torch
import torch.nn as nn
from U_Net_helper_functions import *
import lightning.pytorch as pl

class Convolutional_Block(nn.Module):
    """
    A convolutional block consisting of three Conv2D layers, each followed by 
    Batch Normalization, ReLU activation, and Dropout. A residual connection 
    is added after the second and third convolutions.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolution kernel. Default is 3.
        padding (int, optional): Padding added to all sides of the input. Default is 1.
        stride (int, optional): Stride of the convolution. Default is 1.
        dropout (float, optional): Dropout probability. Default is 0.1.
    """
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
    """
    A U-Net architecture implemented in PyTorch Lightning for image segmentation tasks.

    The model is built using a symmetric encoder-decoder structure:
    - The encoder path applies convolutional blocks followed by max pooling, 
      using the number of channels specified in `list_of_chanel_numbers`.
    - The decoder path reverses this structure using transposed convolutions 
      (upsampling) and corresponding convolutional blocks.

    Args:
        list_of_chanel_numbers (list of int): Defines the number of channels 
            at each level of the U-Net. Each consecutive pair is used to create
            a convolutional block in the encoder, followed by max pooling.
            The reversed list is used in the decoder with transposed convolutions.
        crop (int, optional): Spatial size to crop inputs/labels for evaluation. Default is None.
        loss_fn (nn.Module, optional): Loss function to be used. Default is CrossEntropyLoss.
        lr (float, optional): Learning rate for the optimizer. Default is 0.001.
        betas (tuple of float, optional): Betas for the Adam optimizer. Default is (0.9, 0.999).
        kernel_size (int, optional): Kernel size for all convolutional layers. Default is 3.
        dropout (float, optional): Dropout probability in convolutional blocks. Default is 0.1.
        padding (int, optional): Padding size for convolutional layers. Default is 1.
        log (bool, optional): Whether to log metrics during training/validation. Default is False.
    """
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
            x = module(x)
            residuals.append(x)
            if i < len((self.bottom_path)):
                x = self.pool(x)

        residuals.reverse()
        x = self.bottom_block(x)

        for i, module in enumerate(self.top_path):
            if i >= 0:
                x = self.UpConvs[i](x)
            x = x + residuals[i]
            x = module(x)
            
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
    
    def run_model_on_validation_dataloder(self, val_dataloader, calculate_pixel_wise_accuracy = False, calculate_tpr = False):
        self.eval()
        with torch.no_grad():
            total_loss = 0
            batches = 0
            total_accuracy = 0
            total_tpr = 0
            for data in val_dataloader:
                batches +=1
                inputs, labels = data
                labels = labels.squeeze().long()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

                if calculate_pixel_wise_accuracy == True:
                    output_mask = torch.argmax(outputs, dim = 1)
                    # print(output_mask.shape, labels.shape)
                    accuracy = pixel_wise_accuracy(output_mask, labels)
                    total_accuracy += accuracy

                if calculate_tpr == True:
                    output_mask = torch.argmax(outputs, dim = 1)
                    # print(output_mask.shape, labels.shape)
                    tpr = true_positive_rate(output_mask, labels.squeeze())
                    total_tpr += tpr
                
            print('Avarage validation loss per batch: ', total_loss / batches)
            if calculate_pixel_wise_accuracy == True:
                print('Avarage pixel wise accuracy per batch: ', total_accuracy / batches)
            if calculate_tpr == True:
                print('Avarage true positive rate per batch: ', total_tpr / batches)
            
            return total_loss / batches
    
    # def print_model_outputs(self, output, og = None, label = None):
