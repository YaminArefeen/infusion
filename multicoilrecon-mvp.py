#!/usr/bin/env python
import bart, sigpy as sp, numpy as np, torch, lpips, glob, os, sys, time, random, cfl
import dnnlib, pickle

from models import models
from multiprocessing import Pool

from torch_utils import distributed as dist
from pytorch_wavelets import DTCWTForward, DTCWTInverse, DWTForward,DWTInverse

device = 'cuda:3'

scale_data = 105.63492 

#- selecting a particular slice and scan to work with
dims = 2 # number of dimensions (either 2D or 3D reconstruction)

#- fourier encoding parameters
encoding    = ['gaussian']
numfeatures = [128]
sigma       = [9]

#- neural network parameters
layers     = [4]
activation = ['relu']
hidden_dimension = [256]

#- optimization parameters
learning_rate = [1e-3]
epochs        = [8000]

#- acquired k-space parameters 
Rs = [[6]]
acs_pctg = [[0,0]]

#- other flags
do_diffusions = [1,0]
normalization_flags = [False] # performing batch normalization after each hidden layer
complex_flag = True
lpips_nets = ['vgg']
perceptual_weightings = [1/500]
loss_weights_types = ['linear']
    
#- spatial regularization
spatial_regularizations = [{'reg':'wav','lam':1e-8}] 
#  reg = -1    => no spatial regularization
#  reg = 'wav' => wavelet regularization

print_loss = 100
#- creating parameter structure
parameters = []
for encode in encoding:
    for numfeat in numfeatures:
        for sig in sigma:
            for layer in layers:
                for act in activation:
                    for hidden in hidden_dimension:
                        for lr in learning_rate:
                            for epoch in epochs:
                                for R in Rs:
                                    for acs_pc in acs_pctg:
                                        for normalization_flag in normalization_flags:
                                            for lpip_net in lpips_nets:
                                                for perceptual_weighting in perceptual_weightings:
                                                    for loss_weights_type in loss_weights_types:
                                                        for do_diffusion in do_diffusions:
                                                            for spatial_regularization in spatial_regularizations:
                                                                parameter = {'encoding':encode,
                                                                     'numfeatures':numfeat,
                                                                     'sigmas':sig,
                                                                     'layers':layer,
                                                                     'activation':act,
                                                                     'hidden_dimension':hidden,
                                                                     'learning_rate':lr,
                                                                     'epochs':epoch,
                                                                     'Rs':R,
                                                                     'acs_pctg':acs_pc,
                                                                     'normalization_flag':normalization_flag,
                                                                     'lpip_net':lpip_net,
                                                                     'perceptual_weighting':perceptual_weighting,
                                                                     'loss_weights_type':loss_weights_type,
                                                                     'spatial_regularization':spatial_regularization,
                                                                     'do_diffusion':do_diffusion}
                                                                parameters.append(parameter)
print('total recons: %d'%len(parameters))

def print_parameter(parameter):
    for key in parameter.keys():
        print(key +': {}'.format(parameter[key]))
    return None

#- loading the fully-sampled k-space
x = cfl.readcfl("fastmri-slice-multicoil")

#- calibrating coil maps
coils = bart.bart(1,'ecalib -m 1',x)

#- squeezing everything
x = x.squeeze().transpose((2,0,1)) / scale_data
coils = coils.squeeze().transpose((2,0,1))
gt = np.sum(np.conj(coils)*sp.ifft(x,axes=(-1,-2)),axis=-3) / (np.sum(np.conj(coils)*coils,axis=-3)+1e-12)

#- diffusion model noise parameters
num_steps = 10
sigma_min = 0.002
tmax = 1

minimum_tmin = 0
minimum_diffusion_loss_weight = 0

#- random diffusion parameters that don't matter that much
class_labels = None
class_idx = None
S_noise = 0
rho=7

# loading network and random seeds
if dist.get_rank() != 0:
        torch.distributed.barrier()
        
dist.print0(f'Loading network network-snapshot-010000.pkl...')
with dnnlib.util.open_url("network-snapshot-010000.pkl", verbose=(dist.get_rank() == 0)) as f:
    diffusion_model = pickle.load(f)['ema'].to(device)

C,M,N = x.shape
xcoords = np.linspace(0,1,M)
ycoords = np.linspace(0,1,N)

coords = np.zeros((M,N,dims))
for mm in range(M):
    for nn in range(N):
        coords[mm,nn,0] = xcoords[mm]
        coords[mm,nn,1] = ycoords[nn]
        
coords_vector = coords.reshape(M*N,dims)

def add_acs(mask,acs_percentage):
    res = [mask.shape[1],mask.shape[2]]

    if(np.sum(acs_percentage)): #if the pctg is greater than 0
        acs_size_halfy = np.floor(res[0] * acs_percentage[0] / 2 / 100)
        acs_size_halfz = np.floor(res[1] * acs_percentage[1] / 2 / 100)

        acs_samplesy   = np.arange(res[0]//2 - acs_size_halfy,res[0]//2 + acs_size_halfy,dtype=int)
        acs_samplesz   = np.arange(res[1]//2 - acs_size_halfz,res[1]//2 + acs_size_halfz,dtype=int)

        print(acs_samplesy)
        print(acs_samplesz)
        for yy in acs_samplesy:
            for zz in acs_samplesz:
                mask[...,yy,zz] = 1

    return mask
    
def fft2(x):
    # assume input of dimension ... x M x N
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x, dim=(-2,-1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1)) 
    
    return x

def ifft2(x):
    # assume input of dimension ... x M x N
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x, dim=(-2,-1), norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1)) 
    
    return x

#- wavelet spatial regularization if I need it
xfm = DWTForward(J=3, mode='zero', wave='db3').to(device)

def Wfor(x,W):
    '''
    inputs:
        output (M x N)  - current estimated image from implicit model
        W     a       - pytorch wavelet operator
    '''
    
    wlr,whr = W(torch.view_as_real(x).permute(2,0,1)[None,...])
    l1wavelet_loss = torch.sum(torch.sqrt(wlr[:,0,...]**2+wlr[:,1,...]**2))

    for a_whr in whr:
        l1wavelet_loss += torch.sum(torch.sqrt(a_whr[:,0,...]**2+a_whr[:,1,...]**2))
        
    return l1wavelet_loss

criterion_data       = torch.nn.MSELoss()
criterion_diffusion  = torch.nn.L1Loss()

pad=torch.zeros((1,1,M,N),device=device) #padding for perceptual loss

numparams=[]
all_x_nexts = []
all_losses_total = []
all_losses_data = []
all_losses_spatial = []
all_losses_diffusion = []
all_losses_diffusion_pure = [] # pure diffusion loss recording without any weighting
all_losses_perception = []
all_losses_perception_pure = [] # pure perception loss recording without any weighting
all_tsamples = []
all_noisy_diffusion_starts = []
all_outputs = []
all_ksp = []
all_nets = [] 
Bs = []

bm = np.zeros_like(gt)
bm[np.abs(gt)>0]= 1
    
l2 = lambda x: np.sqrt(np.sum(np.abs(x)**2,axis=(-1,-2)))

for pp,parameter in enumerate(parameters):
    start_time = time.time()
    print('PARMAETER SET %d/%d'%(pp+1,len(parameters)))
    print_parameter(parameter)
    
    do_diffusion = parameter['do_diffusion']
    
    #- setting perceptual loss
    criterion_perception = lpips.LPIPS(net=parameter['lpip_net']).to(device)
    
    #- preparing weightings in optimization
    tmin = torch.linspace(tmax,minimum_tmin,parameter['epochs'],device=device)
        
    # weighting decay function of perceptual loss
    if parameter['loss_weights_type'] == 'linear':
        loss_weights = np.linspace(1,minimum_diffusion_loss_weight,parameter['epochs'])
    elif parameter['loss_weights_type'] == 'exponential':
        time_points  = np.arange(0,parameter['epochs'])
        loss_weights = np.exp(-5*time_points/np.max(time_points))
    elif parameter['loss_weights_type'] == 'log':
        time_points = np.flip(np.linspace(1,np.exp(1),parameter['epochs']))
        loss_weights= np.log(time_points)
    
    #- generating under-sampled k-space
    mask = np.zeros((C,M,N),dtype=complex)
    
    if len(parameter['Rs']) == 1:
        mask[...,:,:] = bart.bart(1,'poisson -Y %d -Z %d -y %.1f -z %.1f'%(M,N,np.sqrt(parameter['Rs']),np.sqrt(parameter['Rs']))).squeeze()
    elif len(parameter['Rs']) == 2:
        mask[...,::parameter['Rs'][0],::parameter['Rs'][1]] = 1
        
    mask = add_acs(mask,parameter['acs_pctg'])
    
    ksp = x*mask  
    all_ksp.append(ksp)
    
    ksp   = torch.tensor(ksp,device=device)
    mask  = torch.tensor(mask,device=device)
    coils = torch.tensor(coils,device=device,dtype=torch.complex128)
    zf = torch.sum(torch.conj(coils)*ifft2(ksp),-3).cpu().detach().numpy()
    
    #- setting up coordinate encoding for implicit
    encoding = parameter['encoding']
    
    if encoding == 'gaussian':
        numfeatures = parameter['numfeatures']
        B = np.random.normal(0, sigma, (numfeatures//2, dims))
        Bs.append(B)
        
        def encode(coordinates, gaussian_matrix):
            coordinates_encoded = np.concatenate((
                np.cos(2 * np.pi * np.matmul(B,coordinates.transpose())),
                np.sin(2 * np.pi * np.matmul(B,coordinates.transpose()))),axis=0).transpose()
            return coordinates_encoded

    else:
        def encode(coordinates, gaussian_matrix):
            return coordinates
        numfeatures = 2
    
    coords_vector_encode = encode(coords_vector,B)
    coords_vector_encode = torch.tensor(coords_vector_encode,
                                    device=device,dtype=torch.float)
    
    #- initializing mlp model, criterion, and optimizer
    net = models.mlp(numfeatures,parameter['hidden_dimension'],
              parameter['layers'],parameter['activation']
                     ,complex_flag,parameter['normalization_flag']).to(device)
    numparams.append(net.count_parameters())
    print('parameter count: %d\ncoordinate count %d' % (numparams[-1],M*N))
    optimizer = torch.optim.Adam(net.parameters(),lr=parameter['learning_rate'])
    
    x_nexts = []
    tsamples = []
    losses_total = []
    losses_data = []
    losses_spatial = []
    losses_diffusion = []
    losses_diffusion_pure = []
    losses_perception = []
    losses_perception_pure = []
    noisy_diffusion_starts = []
    outputs = []
    
    #- attempting to perform reconstruction
    for epoch in range(parameter['epochs']):
        time_epoch_start = time.time()
        
        #- updating with mlp loss with respect to data
        optimizer.zero_grad()

        output = torch.view_as_complex(net(coords_vector_encode)).reshape(M,N)
        
        Ax = fft2(coils*output[None,...]) * mask
        loss_data   = criterion_data(torch.view_as_real(Ax),torch.view_as_real(ksp))
        losses_data.append(loss_data.item())
        
        #- computing a sptial loss if so desired
        if parameter['spatial_regularization']['reg'] == 'wav':
            loss_spatial = Wfor(output,xfm) * parameter['spatial_regularization']['lam']
            losses_spatial.append(loss_spatial.item())
            
        if do_diffusion:
            #- loss with respect to diffusion model
            #     preparing output to be of appropriate dimensions for diffusion
            output_diffusion = torch.view_as_real(output).permute(2,0,1)[None,...] 
            noise = torch.randn([1, 2, M, N], device=device,dtype=output_diffusion.dtype)

            #     sample the noise level
            tsample = torch.rand(1,device=device)[0] * tmax
            if tsample<tmin[epoch]: tsample = tmin[epoch]
            tsamples.append(tsample.cpu().numpy())

            #     performing diffusion process
            step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
            t_steps = (tsample ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - tsample ** (1 / rho))) ** rho
            t_steps = torch.cat([diffusion_model.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

            x_next = output_diffusion + noise * tsample
            x_next = x_next.detach() # detaching for faster computation (don't need gradients)
            
            noisy_diffusion_starts.append(x_next.detach().cpu().numpy().squeeze())

            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
                denoised = diffusion_model(x_next, t_cur, class_labels).to(torch.float32)
                d_cur = (x_next - denoised)/t_cur
                x_next = x_next + (t_next - t_cur) * d_cur

            #- computing structural diffusion loss
            loss_diffusion = criterion_diffusion(output_diffusion,x_next)
            losses_diffusion_pure.append(loss_diffusion.item())
            
            loss_diffusion *= loss_weights[epoch] * (tmax-tsample)/50         
            losses_diffusion.append(loss_diffusion.item())

            #- computing perceptual diffusion loss
            output_diffusion_pad = torch.cat((output_diffusion,pad),dim=1)
            x_next_pad = torch.cat((x_next,pad),dim=1)
            
            loss_perception = criterion_perception.forward(output_diffusion_pad,\
                    x_next_pad)
            losses_perception_pure.append(loss_perception.item())
            
            loss_perception *= loss_weights[epoch]*parameter['perceptual_weighting']
            losses_perception.append(loss_perception.item())
        
        if(epoch % print_loss == 0):
            print('  iteration %d/%d' % (epoch+1,parameter['epochs']))
            if do_diffusion: print('    current diffusion   loss: %.8f' % loss_diffusion.item())
            print('    current consistency loss: %.8f' % loss_data.item())            
            if do_diffusion: print('    current perception  loss: %.8f' % loss_perception.item())
            if parameter['spatial_regularization']['reg'] == 'wav':
                print('    current spatial loss:     %.8f' % loss_spatial.item())            
                
        #- back prop from loss
        loss = loss_data 
        
        if do_diffusion:
            loss = loss + loss_perception
          
        if parameter['spatial_regularization']['reg'] == 'wav':
            loss = loss + loss_spatial
          
        losses_total.append(loss.item())

        if(epoch % print_loss == 0):
            print('    total current loss:        %.8f' % loss.item())    
        
        loss.backward()
        optimizer.step()
      
        if do_diffusion:
            tmp = x_next.detach().cpu().numpy().squeeze()
            x_nexts.append(tmp[0,...]+1j*tmp[1,...])
            
        current_output = output.detach().cpu().numpy()
        if(epoch % print_loss == 0):
            print('    nrmse: %.2f' % (l2(gt-current_output*bm)/l2(gt)*100))
        
        outputs.append(current_output)
        
        time_epoch_end = time.time()
        if(epoch % print_loss == 0):
            print('time per epoch: %.2f' % (time_epoch_end - time_epoch_start))
       
    all_losses_total.append(losses_total)
    all_losses_data.append(losses_data)
    all_losses_spatial.append(losses_spatial)
    all_outputs.append(outputs)
    all_nets.append(net)
    if do_diffusion:
        all_losses_perception.append(losses_perception)
        all_losses_perception_pure.append(losses_perception_pure)
        all_losses_diffusion.append(losses_diffusion)
        all_losses_diffusion_pure.append(losses_diffusion_pure)
        all_x_nexts.append(x_nexts)
        all_noisy_diffusion_starts.append(noisy_diffusion_starts)
        all_tsamples.append(tsamples)
        
    end_time = time.time()

# - saving the different stuff
cfl.writecfl('numparams',np.array(numparams))
cfl.writecfl('all_losses_data',np.array(all_losses_data))
if do_diffusion:
    cfl.writecfl('all_x_nexts',np.array(all_x_nexts))
    cfl.writecfl('all_losses_diffusion',np.array(all_losses_diffusion))
    cfl.writecfl('all_losses_perception',np.array(all_losses_perception))

    cfl.writecfl('all_tsamples',np.array(all_tsamples))
    cfl.writecfl('all_noisy_diffusion_starts',np.array(all_noisy_diffusion_starts))
cfl.writecfl('all_outputs',np.array(all_outputs))
cfl.writecfl('all_ksp',np.array(all_ksp))

cfl.writecfl('truth',np.array(gt))

np.save('do_diffusion',np.array(do_diffusion),allow_pickle=True)
np.save('Bs',Bs,allow_pickle=True)
np.save('parameters',parameters,allow_pickle=True)
