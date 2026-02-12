import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y, train=False, transform=None):

        x = np.array(x)

        min_dim = min(x.shape[-2:])

        if min_dim >= 16:
            cutout_length = min(x.shape[-2:]) // 8
        else:
            cutout_length = 1

        # If input is (h, w), add channel and batch dimensions -> (1, 1, h, w)
        if x.ndim == 2:  # (h, w)
            x = x[None, ...]  # (1, h, w)
        # If input is (c, h, w), add batch dimension -> (1, c, h, w)
        if x.ndim == 3:
            if x.shape[0] <= 4:  # likely (c, h, w)
                x = x[None, ...]  # (1, c, h, w)
            else:  # likely (n, h, w)
                x = x[:, None, ...]  # (n, 1, h, w)
        # If input is (n, h, w, c), move channels to second dim -> (n, c, h, w)
        if x.ndim == 4 and x.shape[1] not in [1, 3]:
            if x.shape[-1] in [1, 3]:
                x = np.moveaxis(x, -1, 1)

        self.x = x
        self.y = y

        

        self.mean = torch.mean(torch.tensor(x), [0, 2, 3])  # Compute mean
        self.std = torch.std(torch.tensor(x), [0, 2, 3])    # Compute std
            
        if train:
            self.transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)#,
                #Cutout(cutout_length)  
            ])  
                
                
        else:            
            
            self.transform = transforms.Compose([
                transforms.Normalize(self.mean, self.std)  
            ])      
            
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        im = torch.tensor(self.x[idx], dtype=torch.float32)

        if self.transform is not None:
            im = self.transform(im)

        if self.y is None:
            return im
        else:
            return im, torch.tensor(self.y[idx], dtype=torch.long)

class DataProcessor:
    """
    -===================================================================================================================
    INIT ===============================================================================================================
    ====================================================================================================================
    The DataProcessor class will receive the following inputs:
        * train_x: numpy array of shape [n_train_datapoints, channels, height, width], these are the training inputs
        * train_y: numpy array of shape [n_train_datapoints], these are the training labels
        * valid_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the validation inputs
        * valid_y: numpy array of shape [n_valid_datapoints], these are the validation labels
        * test_x: numpy array of shape [n_valid_datapoints, channels, height, width], these are the test inputs
        * metadata: A dictionary with information about this dataset, with the following keys:
            'num_classes' : The number of output classes in the classification problem
            'codename' : A unique string that represents this dataset
            'input_shape': A tuple describing [n_total_datapoints, channel, height, width] of the input data
            'time_remaining': The amount of compute time left for your submission

    You can modify or add anything into the metadata that you wish, if you want to pass messages between your classes

    """
    def __init__(self, train_x, train_y, valid_x, valid_y, test_x, metadata, clock):

        def resize_if_needed(arr, name):
        
            max_dim = 128 
            min_dim = 8
            arr = np.array(arr)
            # Only add channel dimension if missing
            if len(arr.shape) == 3:
                arr = arr[:, None, :, :]
            # Now safe to check spatial dims
            C = arr.shape[1]
            H = arr.shape[2]
            W = arr.shape[3]
            
            print('Input Dims: ',C ,' x ', H, ' x ', W)            
            
            if max(H,W) > max_dim:
                if (H > W):
                    H = max_dim
                elif (W > H):
                    W = max_dim
                else:
                    H = max_dim
                    W = max_dim  
  
      
            if min(H,W) < min_dim:
                if (H < W):
                    H = min_dim
                elif (W < H):
                    W = min_dim
                else:
                    H = min_dim
                    W = min_dim  
            
            print('Output Dims: ', C ,' x ', H, ' x ', W)                  

            arr = torch.tensor(arr)
            arr = F.interpolate(arr, size=(H, W), mode='bilinear', align_corners=False).numpy()
                         
            return arr

        self.train_x = resize_if_needed(train_x, 'train_x')
        self.train_y = train_y
        self.valid_x = resize_if_needed(valid_x, 'valid_x')
        self.valid_y = valid_y
        self.test_x = resize_if_needed(test_x, 'test_x')
        self.metadata = metadata
        self.metadata["input_shape"] = list(self.train_x.shape)

    """
    ====================================================================================================================
    PROCESS ============================================================================================================
    ====================================================================================================================
    This function will be called, and it expects you to return three outputs:
        * train_loader: A Pytorch dataloader of (input, label) tuples
        * valid_loader: A Pytorch dataloader of (input, label) tuples
        * test_loader: A Pytorch dataloader of (inputs)  <- Make sure shuffle=False and drop_last=False!
        
    See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader for more info.  
        
    Here, you can do whatever you want to the input data to process it for your NAS algorithm and training functions
    """
    def process(self):
        # create train, valid, and test datasets
        train_ds = Dataset(self.train_x, self.train_y, train=True)
        valid_ds = Dataset(self.valid_x, self.valid_y, train=False)
        test_ds = Dataset(self.test_x, None, train=False)

        batch_size = 64

        # build data loaders
        train_loader = torch.utils.data.DataLoader(train_ds,
                                                   batch_size=batch_size,
                                                   drop_last=True,
                                                   shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_ds,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_ds,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  drop_last=False)
        return train_loader, valid_loader, test_loader
