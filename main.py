# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:09:29 2023

@author: varsh
"""
import os
import math
import torch
import torch.nn as nn
import torch.fft

import matplotlib.pyplot as plt 
import pickle
import imageio

import argparse
import random
import numpy as np
from imageio import imread
from skimage.transform import resize

import utils

class ImageLoaderModified:
    """Loads images a folder with augmentation for generator training

    Class initialization parameters
    -------------------------------
    data_path: folder containing images
    channel: color channel to load (0, 1, 2 for R, G, B, None for all 3),
        default None
    batch_size: number of images to pass each iteration, default 1
    image_res: 2d dimensions to pad/crop the image to for final output, default
        (1080, 1920)
    homography_res: 2d dims to scale the image to before final crop to image_res
        for consistent resolutions (crops to preserve input aspect ratio),
        default (880, 1600)
    shuffle: True to randomize image order across batches, default True
    vertical_flips: True to augment with vertical flipping, default True
    horizontal_flips: True to augment with horizontal flipping, default True
    idx_subset: for the iterator, skip all but these images. Given as a list of
        indices corresponding to sorted filename order. Forces shuffle=False and
        batch_size=1. Defaults to None to not subset at all.
    crop_to_homography: if True, only crops the image instead of scaling to get
        to target homography resolution, default False

    Usage
    -----
    To be used as an iterator:

    >>> image_loader = ImageLoader(...)
    >>> for ims, input_resolutions, filenames in image_loader:
    >>>     ...

    ims: images in the batch after transformation and conversion to linear
        amplitude, with dimensions [batch, channel, height, width]
    input_resolutions: list of length batch_size containing tuples of the
        original image height/width before scaling/cropping
    filenames: list of input image filenames, without extension

    Alternatively, can be used to manually load a single image:

    >>> ims, input_resolutions, filenames = image_loader.load_image(idx)

    idx: the index for the image to load, indices are alphabetical based on the
        file path.
    """

    def __init__(self, data_path, channel=None, batch_size=1,
                 image_res=(1080, 1920),
                 shuffle=True, vertical_flips=True, horizontal_flips=True,
                 idx_subset=None):
        if not os.path.isdir(data_path):
            raise NotADirectoryError(f'Data folder: {data_path}')
        self.data_path = data_path
        self.channel = channel
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_res = image_res
        self.subset = idx_subset

        self.augmentations = []
        if vertical_flips:
            self.augmentations.append(self.augment_vert)
        if horizontal_flips:
            self.augmentations.append(self.augment_horz)
        # store the possible states for enumerating augmentations
        self.augmentation_states = [fn() for fn in self.augmentations]

        self.im_names = get_image_filenames(data_path)
        self.im_names.sort()

        # if subsetting indices, force no randomization and batch size 1
        if self.subset is not None:
            self.shuffle = False
            self.batch_size = 1

        # create list of image IDs with augmentation state
        self.order = ((i,) for i in range(len(self.im_names)))
        for aug_type in self.augmentations:
            states = aug_type()  # empty call gets possible states
            # augment existing list with new entry to states tuple
            self.order = ((*prev_states, s)
                          for prev_states in self.order
                          for s in states)
        self.order = list(self.order)

    def __iter__(self):
        self.ind = 0
        if self.shuffle:
            random.shuffle(self.order)
        return self

    def __next__(self):
        if self.subset is not None:
            while self.ind not in self.subset and self.ind < len(self.order):
                self.ind += 1

        if self.ind < len(self.order):
            batch_ims = self.order[self.ind:self.ind+self.batch_size]
            self.ind += self.batch_size
            return self.load_batch(batch_ims)
        else:
            raise StopIteration

    def __len__(self):
        if self.subset is None:
            return len(self.order)
        else:
            return len(self.subset)

    def load_batch(self, images):
        im_res_name = [self.load_image(*im_data) for im_data in images]
        ims = torch.stack([im for im, _, _ in im_res_name], 0)
        return (ims,
                [res for _, res, _ in im_res_name],
                [name for _, _, name in im_res_name])

    def load_image(self, filenum, *augmentation_states):
        im = imread(self.im_names[filenum])
        im = resize(im, image_res)

        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)  # augment channels for gray images

        if self.channel is None:
            im = im[..., :3]  # remove alpha channel, if any
        else:
            # select channel while keeping dims
            im = im[..., self.channel, np.newaxis]

        im = utils.im2float(im, dtype=np.float64)  # convert to double, max 1

        # linearize intensity and convert to amplitude
        low_val = im <= 0.04045
        im[low_val] = 25 / 323 * im[low_val]
        im[np.logical_not(low_val)] = ((200 * im[np.logical_not(low_val)] + 11)
                                       / 211) ** (12 / 5)
        im = np.sqrt(im)  # to amplitude

        # move channel dim to torch convention
        im = np.transpose(im, axes=(2, 0, 1))

        # apply data augmentation
        for fn, state in zip(self.augmentations, augmentation_states):
            im = fn(im, state)

        # normalize resolution
        input_res = im.shape[-2:]
        # im = resize_keep_aspect(im, self.image_res)
        # im = pad_crop_to_res(im, self.image_res)

        return (torch.from_numpy(im).float(),
                input_res,
                os.path.splitext(self.im_names[filenum])[0])

    def augment_vert(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1, :]
        return image

    def augment_horz(self, image=None, flip=False):
        if image is None:
            return (True, False)  # return possible augmentation values

        if flip:
            return image[..., ::-1]
        return image


def get_image_filenames(dir):
    """Returns all files in the input directory dir that are images"""
    image_types = ('jpg', 'jpeg', 'tiff', 'tif', 'png', 'bmp', 'gif')
    files = os.listdir(dir)
    exts = (os.path.splitext(f)[1] for f in files)
    images = [os.path.join(dir, f)
              for e, f in zip(exts, files)
              if e[1:] in image_types]
    return images


def resize_keep_aspect(image, target_res, pad=False):
    """Resizes image to the target_res while keeping aspect ratio by cropping

    image: an 3d array with dims [channel, height, width]
    target_res: [height, width]
    pad: if True, will pad zeros instead of cropping to preserve aspect ratio
    """
    im_res = image.shape[-2:]

    # finds the resolution needed for either dimension to have the target aspect
    # ratio, when the other is kept constant. If the image doesn't have the
    # target ratio, then one of these two will be larger, and the other smaller,
    # than the current image dimensions
    resized_res = (int(np.ceil(im_res[1] * target_res[0] / target_res[1])),
                   int(np.ceil(im_res[0] * target_res[1] / target_res[0])))

    # only pads smaller or crops larger dims, meaning that the resulting image
    # size will be the target aspect ratio after a single pad/crop to the
    # resized_res dimensions
    if pad:
        image = utils.pad_image(image, resized_res, pytorch=False)
    else:
        image = utils.crop_image(image, resized_res, pytorch=False)

    # switch to numpy channel dim convention, resize, switch back
    image = np.transpose(image, axes=(1, 2, 0))
    image = resize(image, target_res, mode='reflect')
    return np.transpose(image, axes=(2, 0, 1))


def pad_crop_to_res(image, target_res):
    """Pads with 0 and crops as needed to force image to be target_res

    image: an array with dims [..., channel, height, width]
    target_res: [height, width]
    """
    return utils.crop_image(utils.pad_image(image,
                                            target_res, pytorch=False),
                            target_res, pytorch=False)


def propagation_ASM(u_in, feature_size, wavelength, z, linear_conv=True,
                    padtype='zero', return_H=False, precomped_H=None,
                    return_H_exp=False, precomped_H_exp=None,
                    dtype=torch.float32):
    """Propagates the input field using the angular spectrum method

    Inputs
    ------
    u_in: PyTorch Complex tensor (torch.cfloat) of size (num_images, 1, height, width) -- updated with PyTorch 1.7.0
    feature_size: (height, width) of individual holographic features in m
    wavelength: wavelength in m
    z: propagation distance
    linear_conv: if True, pad the input to obtain a linear convolution
    padtype: 'zero' to pad with zeros, 'median' to pad with median of u_in's
        amplitude
    return_H[_exp]: used for precomputing H or H_exp, ends the computation early
        and returns the desired variable
    precomped_H[_exp]: the precomputed value for H or H_exp
    dtype: torch dtype for computation at different precision

    Output
    ------
    tensor of size (num_images, 1, height, width, 2)
    """

    if linear_conv:
        # preprocess with padding for linear conv.
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        if padtype == 'zero':
            padval = 0
        elif padtype == 'median':
            padval = torch.median(torch.pow((u_in**2).sum(-1), 0.5))
        u_in = utils.pad_image(u_in, conv_size, padval=padval, stacked_complex=False)

    if precomped_H is None and precomped_H_exp is None:
        # resolution of input field, should be: (num_images, num_channels, height, width, 2)
        field_resolution = u_in.size()

        # number of pixels
        num_y, num_x = field_resolution[2], field_resolution[3]

        # sampling inteval size
        dy, dx = feature_size

        # size of the field
        y, x = (dy * float(num_y), dx * float(num_x))

        # frequency coordinates sampling
        fy = np.linspace(-1 / (2 * dy) + 0.5 / (2 * y), 1 / (2 * dy) - 0.5 / (2 * y), num_y)
        fx = np.linspace(-1 / (2 * dx) + 0.5 / (2 * x), 1 / (2 * dx) - 0.5 / (2 * x), num_x)

        # momentum/reciprocal space
        FX, FY = np.meshgrid(fx, fy)

        # transfer function in numpy (omit distance)
        HH = 2 * math.pi * np.sqrt(1 / wavelength**2 - (FX**2 + FY**2))

        # create tensor & upload to device (GPU)
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)

        ###
        # here one may iterate over multiple distances, once H_exp is uploaded on GPU

        # reshape tensor and multiply
        H_exp = torch.reshape(H_exp, (1, 1, *H_exp.size()))

    # handle loading the precomputed H_exp value, or saving it for later runs
    elif precomped_H_exp is not None:
        H_exp = precomped_H_exp

    if precomped_H is None:
        # multiply by distance
        H_exp = torch.mul(H_exp, z)

        # band-limited ASM - Matsushima et al. (2009)
        fy_max = 1 / np.sqrt((2 * z * (1 / y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1 / x))**2 + 1) / wavelength
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)

        # get real/img components
        H_real, H_imag = utils.polar_to_rect(H_filter.to(u_in.device), H_exp)

        H = torch.stack((H_real, H_imag), 4)
        H = utils.ifftshift(H)
        H = torch.view_as_complex(H)
    else:
        H = precomped_H

    # return for use later as precomputed inputs
    if return_H_exp:
        return H_exp
    if return_H:
        return H

    # For who cannot use Pytorch 1.7.0 and its Complex tensors support:
    # # angular spectrum
    # U1 = torch.fft(utils.ifftshift(u_in), 2, True)
    #
    # # convolution of the system
    # U2 = utils.mul_complex(H, U1)
    #
    # # Fourier transform of the convolution to the observation plane
    # u_out = utils.fftshift(torch.ifft(U2, 2, True))

    U1 = torch.fft.fftn(utils.ifftshift(u_in), dim=(-2, -1), norm='ortho')

    U2 = H * U1

    u_out = utils.fftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))

    if linear_conv:
        # return utils.crop_image(u_out, input_resolution) # using stacked version
        return utils.crop_image(u_out, input_resolution, pytorch=True, stacked_complex=False)  # using complex tensor
    else:
        return u_out
    
# Creating a DeepEncoder class 
class DeepEncoder(nn.Module):
    def __init__(self, device, inputSize = [1920,1080]):
        super(DeepEncoder, self).__init__()

        self.layers = nn.ModuleList();
        inpSize = inputSize[0] * inputSize[1];  
        numLayers = 3
        current_dim = inpSize;
        numNeuronsPerLyr = int(inpSize/2)
        outputDim= inpSize;
        
        for lyr in range(numLayers): # define the layers
            l = nn.Linear(current_dim, numNeuronsPerLyr,device=device,dtype=torch.float32);
            nn.init.xavier_normal_(l.weight);
            nn.init.zeros_(l.bias);
            self.layers.append(l);
            current_dim = numNeuronsPerLyr;
        self.layers.append(nn.Linear(current_dim, outputDim, device=device,dtype=torch.float32));

  
    def forward(self, x): 
        return self.evaluate(x);
        
    def evaluate(self, x):
        m = nn.ReLU();
        nnTanh = nn.Tanh();
        x = x.reshape(-1, image_res[0]*image_res[1])
        ctr = 0;
        for layer in self.layers[:-1]: # forward prop
            x = m((layer(x)));
            ctr += 1;
        encoded = np.pi*nnTanh(self.layers[-1](x)); # -pi to pi output
        
        return encoded
        

class Decoder():
    def __init__(self, dtype,device, propagator, prop_dist, wavelength, feature_size, prop_model, precomputed_H,inputSize = [1920,1080]):
        self.propagator = propagator;
        self.prop_dist = prop_dist;
        self.wavelength = wavelength;
        self.feature_size = feature_size;
        self.prop_model = prop_model;
        self.precomputed_H = precomputed_H;
        self.inputSize = inputSize;
        self.dtype = dtype;
        self.device = device;

    def Decode(self,encoded):
        encoded_reshaped = torch.reshape(encoded, (encoded.shape[0], 1, *self.inputSize))

        real, imag = utils.polar_to_rect_just_ang(encoded_reshaped);
        slm_field = torch.complex(real, imag)
    
        recon_field = utils.propagate_field(slm_field, self.propagator, self.prop_dist, self.wavelength, self.feature_size,
                                            self.prop_model, self.dtype, self.precomputed_H)
    
        # get amplitude
        recon_amp = recon_field.abs()
    
        # crop roi
        recon_amp = utils.crop_image(recon_amp, target_shape=self.inputSize, stacked_complex=False)
    
    
        out_amp = recon_amp
    
        # calculate loss and backprop
        #lossValue = loss(s * out_amp, target_amp)
        return out_amp

class EvaluateTest:
    def __init__(self, encoder, decoders):
        self.encoder = encoder;
        self.decoders = decoders;
        
    def evaluate(self, img):
        out1 = self.encoder.evaluate(img.reshape(img.shape[0]*img.shape[1],1,img.shape[-2],img.shape[-1]))
        holograms = out1.reshape(img.shape[0],img.shape[1],img.shape[-2],img.shape[-1])
        out2List = []
        for i in range(img.shape[1]):
            out2 = decoders[i].Decode(holograms[:,i,:,:]);
            out2List.append(out2)
        out2 = torch.cat(out2List,dim = 1);
        regeneratedImages = torch.mul(out2,torch.div(torch.sum(torch.mul(out2,img),(-2,-1),keepdim=True),torch.sum(torch.mul(out2,out2),(-2,-1),keepdim=True)));
        
        return holograms.cpu().detach().numpy().transpose(2,3,1,0),regeneratedImages.cpu().detach().numpy().transpose(2,3,1,0)
  
def getImages(image_loader, device):
    targetAmps = []
    for k, target in enumerate(image_loader):
        target_amp, _, _ = target
        targetAmps.append(target_amp);

    allAmps = torch.cat(targetAmps, dim = 0);
    allAmps = allAmps.to(device)
    return allAmps;      
        
def saveImages(save_path, holograms, regeneratedImages):
    
    print("Saving final holograms");
    try:
        holoFinal = holograms;#.cpu().detach().numpy();
        for i in range(holoFinal.shape[3]):
            for j in range(holoFinal.shape[2]):
                holoFinalThis = np.squeeze(holoFinal[:,:,j,i]);
                recon_srgb = utils.srgb_lin2gamma(np.clip(holoFinalThis**2, 0.0, 1.0))
                imageio.imwrite(save_path+"holofinal"+str(i) + "ch_" + str(j)+".png", (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))
        print("Saving final holograms successful")
    except:
        print("Saving final holograms not successful")
        
    #allImages[channel] = imgFinal;
    

    print("Saving final images");
    try:
        imgFinal = regeneratedImages;#.cpu().detach().numpy();
        for i in range(imgFinal.shape[3]):
            finalImage = np.squeeze(imgFinal[:,:,:,i]);
            # for channel in imgFinal.shape[1]:
            #     finalImage[:,:,channel] = np.squeeze(allImages[channel][i]);
            recon_srgb = utils.srgb_lin2gamma(np.clip(finalImage**2, 0.0, 1.0))
            imageio.imwrite(save_path+"RGBfinal" + str(i) + ".png", (recon_srgb * np.iinfo(np.uint8).max).round().astype(np.uint8))
        print("Saving final images successful")
    except:
        print("Saving final images not successful")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Run encoder")

parser.add_argument('--usegpu',type=str2bool,nargs='?',const=True,default=True, help='use gpu')
args = parser.parse_args()

data_path = './data'
factor = 10;
cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9
channels = [0,1,2]
prop_dists = (20 * cm, 20 * cm, 20 * cm)
wavelenghts = (638 * nm, 520 * nm, 450 * nm)
feature_size = (6.4*factor * um, 6.4*factor * um)  # SLM pitch
#slm_res = (1080*factor, 1920*factor)  # resolution of SLM
image_res = (int(1080/factor), int(1920/factor))
roi_res = (int(1080/factor), int(1920/factor))#(int(880/factor), int(1600/factor))  # regions of interest (to penalize for SGD)
dtype = torch.float32  # default datatype (Note: the result may be slightly different if you use float64, etc.)
device = torch.device("cuda" if torch.cuda.is_available() and args.usegpu else "cpu")  # The gpu you are using
print("using device = ",device)
propagator = propagation_ASM
prop_model = 'ASM'
roiStart = tuple(int((image_res[i]-roi_res[i])/2) for i in range(len(image_res)));
roiEnd = tuple(int(image_res[i]-roiStart[i]) for i in range(len(image_res)));

criterion = torch.nn.MSELoss() 

# Instantiating the model and hyperparameters 
model = DeepEncoder(device,inputSize = image_res) 
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
allImages = {}

saveDir = "D:" + os.sep + "Extras"+os.sep;

precomputed_Hs = []
decoders = []

for channel in channels:
    targetAmpsChannel = []
    prop_dist = prop_dists[channel]  # propagation distance from SLM plane to target plane
    wavelength = wavelenghts[channel]  # wavelength of each color
    
    precomputed_H = propagator(torch.empty((1,1,*image_res), dtype=torch.float32), feature_size,
                                   wavelength, prop_dist, return_H=True)
    precomputed_H = precomputed_H.to(device).detach()
    precomputed_H.requires_grad = False
    precomputed_Hs.append(precomputed_H);
    decoders.append(Decoder(dtype,device, propagator, prop_dist, wavelength, feature_size, prop_model, precomputed_H,inputSize = image_res)) 

# Augmented image loader (if you want to shuffle, augment dataset, put options accordingly.)

image_loader = ImageLoaderModified(data_path, channel=None,
                           image_res=image_res,
                           shuffle=False, vertical_flips=False, horizontal_flips=False)

allAmps = getImages(image_loader, device).requires_grad_(True);
        
    
#fftOut = torch.fft.fftn(utils.ifftshift(allAmps), dim=(-2, -1), norm='ortho').abs()
batch_size = allAmps.shape[0]*allAmps.shape[1];
        #target_amp = target_amp.to(device)
num_epochs = 500;
# List that will store the training loss 
train_loss = [] 

# Dictionary that will store the 
# different images and outputs for 
# various epochs 
outputs = {} 
saveEvery = 10;
# Training loop starts 
for epoch in range(num_epochs): 
    # loss 
    running_loss =0; print(epoch);
    

    img = allAmps;
    
    out1 = model(img.reshape(img.shape[0]*img.shape[1],1,img.shape[-2],img.shape[-1]))
    out1 = out1.reshape(img.shape[0],img.shape[1],img.shape[-2],img.shape[-1])
    out2List = []
    for i in range(img.shape[1]):
        out2 = decoders[i].Decode(out1[:,i,:,:]);
        out2List.append(out2)
    out2 = torch.cat(out2List,dim = 1);
    out2 = torch.mul(out2,torch.div(torch.sum(torch.mul(out2,img),(-2,-1),keepdim=True),torch.sum(torch.mul(out2,out2),(-2,-1),keepdim=True)));
    out2Roi = out2[:,:,roiStart[0]:roiEnd[0],roiStart[1]:roiEnd[1]]
    imgRoi = img[:,:,roiStart[0]:roiEnd[0],roiStart[1]:roiEnd[1]]
    
    # Calculating loss 
    loss = criterion(out2Roi, imgRoi) 
    
    # Updating weights according 
    # to the calculated loss 
    optimizer.zero_grad() 
    loss.backward(retain_graph=True) 
    optimizer.step() 
    
    # Incrementing loss 
    running_loss += loss.item() 
    print(running_loss)
        
    train_loss.append(running_loss);
    
    # Storing useful images and 
    # reconstructed outputs for the last batch 
    if (epoch%saveEvery == 0 or epoch+1 == num_epochs):
        outputs[epoch+1] = {'outHolo': out1.cpu().detach().numpy().transpose(2,3,1,0),
                            'outImg': out2.cpu().detach().numpy().transpose(2,3,1,0)} ;

save_path = saveDir ;#+ "channel" + str(channel)

# Plotting the training loss 
plt.plot(range(1,num_epochs+1),train_loss) 
plt.xlabel("Number of epochs") 
plt.ylabel("Training Loss") 
plt.savefig(save_path+"convergence.png")
plt.close();
    
print("Saving variables...")
try:
    with open(save_path+"outputs.pkl","wb") as f:
        toSave = [train_loss, outputs];
        pickle.dumps(toSave);
    print("Saving variables successful")
except:
    print("Saving variables not successful")
    
saveImages(save_path, outputs[num_epochs]['outHolo'], outputs[num_epochs]['outImg']);
#print("Saving model...")
#model_path = save_path + "model"
#torch.save(model.state_dict(), model_path)
    
evalTest = EvaluateTest(model, decoders);
image_loader_test = ImageLoaderModified(data_path + os.sep + "Test", channel=None,
                           image_res=image_res,
                           shuffle=False, vertical_flips=False, horizontal_flips=False);
allAmps = getImages(image_loader_test, device);

testHolo, testImg = evalTest.evaluate(allAmps);
saveImages(save_path + os.sep + 'TestResult' + os.sep, testHolo, testImg);