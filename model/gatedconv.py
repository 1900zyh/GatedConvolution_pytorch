import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from core.spectral_norm import use_spectral_norm


class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()
  
  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
    self.apply(init_func)
    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)


def get_pad(in_,  ksize, stride, atrous=1):
  out_ = np.ceil(float(in_)/stride)
  return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

class GatedConv(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, activation=nn.LeakyReLU(0.2, inplace=True)):
    super(GatedConv, self).__init__()
    self.activation = activation
    if out_channels == 3 or activation is None:
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
    else:
      self.conv = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride, padding, dilation)
      self.gating = nn.Conv2d(in_channels, out_channels//2, kernel_size, stride, padding, dilation)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    if self.activation is None:
      return self.conv(x)
    else:
      return self.activation(self.conv(x)) * self.sigmoid(self.gating(x))


class GatedDeConv(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, scale_factor=2, padding=0, dilation=1, 
               activation=torch.nn.LeakyReLU(0.2, inplace=True)):
    super(GatedDeConv, self).__init__()
    self.conv2d = GatedConv(in_channels, out_channels, kernel_size, stride, padding, dilation, activation)
    self.scale_factor = scale_factor

  def forward(self, x):
    x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
    return self.conv2d(x)


class ContextualAttention(nn.Module):
  def __init__(self, ksize=3, stride=1, rate=2, softmax_scale=10., fuse_k=3):
    super(ContextualAttention, self).__init__()
    self.ksize = ksize
    self.stride = stride
    self.rate = rate 
    self.fuse_k = fuse_k
    self.softmax_scale = softmax_scale
    
  def forward(self, x1, x2, mask=None):
    # get shapes
    x1s = list(x1.size())

    # extract patches from low-level feature maps x1 with stride and rate
    kernel = 2*self.rate
    raw_w = extract_patches(x1, kernel=kernel, stride=self.rate*self.stride)
    raw_w = raw_w.contiguous().view(x1s[0], -1, x1s[1], kernel, kernel) # B*HW*C*K*K 
    raw_w_groups = torch.split(raw_w, 1, dim=0) 

    # split high-level feature maps x2 for matching 
    x2 = F.interpolate(x2, scale_factor=1./2, mode='nearest')
    x2s = list(x2.size())
    f_groups = torch.split(x2, 1, dim=0) 
    # extract patches from x2 as weights of filter
    w = extract_patches(x2, kernel=self.ksize, stride=self.stride)
    w = w.contiguous().view(x2s[0], -1, x2s[1], self.ksize, self.ksize) # B*HW*C*K*K
    w_groups = torch.split(w, 1, dim=0) 

    # process mask
    if mask is not None:
      mask = F.interpolate(mask, size=x2s[2:4], mode='bilinear', align_corners=True)
    else:
      mask = torch.zeros([1, 1, x2s[2], x2s[3]])
      if torch.cuda.is_available():
        mask = mask.cuda()
    # extract patches from masks to mask out hole-patches for matching 
    m = extract_patches(mask, kernel=self.ksize, stride=self.stride)
    m = m.contiguous().view(x2s[0], -1, 1, self.ksize, self.ksize)  # B*HW*1*K*K
    m = m.mean([2,3,4]).unsqueeze(-1).unsqueeze(-1)
    mm = m.eq(0.).float() # (B, HW, 1, 1)       
    mm_groups = torch.split(mm, 1, dim=0)

    y = []
    scale = self.softmax_scale
    padding = 0 if self.ksize==1 else 1
    k = self.fuse_k 
    fuse_weight = torch.eye(k).view(1,1,k,k)
    if torch.cuda.is_available():
      fuse_weight = fuse_weight.cuda()
    for xi, wi, raw_wi, mi in zip(f_groups, w_groups, raw_w_groups, mm_groups):
      wi = wi[0]
      escape_NaN = torch.FloatTensor([1e-4])
      if torch.cuda.is_available():
        escape_NaN = escape_NaN.cuda()
      wi_normed = wi / torch.max(torch.sqrt((wi*wi).sum([1,2,3],keepdim=True)), escape_NaN)
      yi = F.conv2d(xi, wi_normed, stride=1, padding=padding)

      # fuse 
      yi = yi.view(1, 1, x2s[2]*x2s[3], x2s[2]*x2s[3])  # (B=1, I=1, H=32*32, W=32*32)
      yi = same_padding(yi, [k, k], [1, 1], [1, 1])
      yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
      yi = yi.contiguous().view(1, x2s[2], x2s[3], x2s[2], x2s[3])  # (B=1, 32, 32, 32, 32)
      yi = yi.permute(0, 2, 1, 4, 3)
      yi = yi.contiguous().view(1, 1, x2s[2]*x2s[3], x2s[2]*x2s[3])
      yi = same_padding(yi, [k, k], [1, 1], [1, 1])
      yi = F.conv2d(yi, fuse_weight, stride=1)
      yi = yi.contiguous().view(1, x2s[3], x2s[2], x2s[3], x2s[2])
      yi = yi.permute(0, 2, 1, 4, 3).contiguous()

      # apply softmax to obtain 
      yi = yi.contiguous().view(1, x2s[2]//self.stride*x2s[3]//self.stride, x2s[2], x2s[3]) 
      yi = yi * mi 
      yi = F.softmax(yi*scale, dim=1)
      yi = yi * mi
      yi = yi.clamp(min=1e-8)

      # attending 
      wi_center = raw_wi[0]
      yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.
      y.append(yi)
    y = torch.cat(y, dim=0)
    y.contiguous().view(x1s)
    return y


# extract patches
def extract_patches(x, kernel=3, stride=1):
  if kernel != 1:
    x = nn.ZeroPad2d(1)(x)
  x = x.permute(0, 2, 3, 1)
  all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
  return all_patches


class InpaintGenerator(nn.Module):
  def __init__(self, ncin=5):
    super(InpaintGenerator, self).__init__()
    cnum = 48
    self.coarse_net = nn.Sequential(
      GatedConv(ncin, cnum, 5, 1, padding=2),
      GatedConv(cnum//2, 2*cnum, 3, 2, padding=1),
      GatedConv(cnum, 2*cnum, 3, 1, padding=1),
      GatedConv(cnum, 4*cnum, 3, 2, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=2, dilation=2),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=4, dilation=4),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=8, dilation=8),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=16, dilation=16),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedDeConv(2*cnum, 2*cnum, 3, padding=1),
      GatedConv(cnum, 2*cnum, 3, 1, padding=1),
      GatedDeConv(cnum, cnum, 3, padding=1),
      GatedConv(cnum//2, cnum//2, 3, 1, padding=1),
      GatedConv(cnum//4, 3, 3, 1, padding=1, activation=None),
      nn.Tanh()
    )

    self.refine_conv = nn.Sequential(
      GatedConv(ncin, cnum, 5, 1, padding=2),
      GatedConv(cnum//2, 2*cnum, 3, 2, padding=1),
      GatedConv(cnum, 2*cnum, 3, 1, padding=1),
      GatedConv(cnum, 4*cnum, 3, 2, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=2, dilation=2),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=4, dilation=4),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=8, dilation=8),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=16, dilation=16),
    )

    self.refine_atn1 = nn.Sequential(
      GatedConv(ncin, cnum, 5, 1, padding=2),
      GatedConv(cnum//2, 2*cnum, 3, 2, padding=1),
      GatedConv(cnum, 2*cnum, 3, 1, padding=1),
      GatedConv(cnum, 4*cnum, 3, 2, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1)
    )
    self.refine_atn2 = ContextualAttention()
    self.refine_atn3 = nn.Sequential(
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
    )

    self.refine_decoder = nn.Sequential(
      GatedConv(4*cnum, 4*cnum, 3, 1, padding=1),
      GatedConv(2*cnum, 4*cnum, 3, 1, padding=1),
      GatedDeConv(2*cnum, 2*cnum, 3, 1, padding=1),
      GatedConv(cnum, 2*cnum, 3, 1, padding=1),
      GatedDeConv(cnum, cnum, 3, 1, padding=1),
      GatedConv(cnum//2, cnum//2, 3, 1, padding=1),
      GatedConv(cnum//4, 3, 3, 1, padding=1, activation=None),
      nn.Tanh()
    )

  def forward(self, x, masks):
    inputs = torch.cat([x, masks, torch.full_like(masks, 1.)], dim=1)
    coarse_x = self.coarse_net(inputs)
    x = x * (1 - masks) + coarse_x * masks
    inputs = torch.cat([x, masks, torch.full_like(masks, 1.)], dim=1)
    conv_x = self.refine_conv(inputs)
    atn_x1 = self.refine_atn1(inputs)
    atn_x2 = self.refine_atn2(atn_x1, atn_x1, masks)
    atn_x3 = self.refine_atn3(atn_x2)
    output = self.refine_decoder(torch.cat([conv_x, atn_x3], dim=1))
    return coarse_x, output


class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid  
    cnum = 64
    self.conv = nn.Sequential(
      use_spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=cnum, kernel_size=5, stride=2, padding=1, bias=not use_sn), use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      use_spectral_norm(nn.Conv2d(in_channels=cnum, out_channels=cnum*2, kernel_size=5, stride=2, padding=1, bias=not use_sn), use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      use_spectral_norm(nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4, kernel_size=5, stride=2, padding=1, bias=not use_sn), use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*4, kernel_size=5, stride=1, padding=1, bias=not use_sn), use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*4, kernel_size=5, stride=1, padding=1, bias=not use_sn), use_sn),
      nn.LeakyReLU(0.2, inplace=True),
      use_spectral_norm(nn.Conv2d(in_channels=cnum*4, out_channels=cnum*4, kernel_size=5, stride=1, padding=1, bias=not use_sn), use_sn),
    )

    if init_weights:
      self.init_weights()
    
  def forward(self, x):
    x = self.conv(x)
    if self.use_sigmoid:
      x = torch.sigmoid(x)
    return x

