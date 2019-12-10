import os
import cv2
import io
import sys
import glob
import time
import zipfile
import subprocess
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from contextlib import contextmanager
import torch
import torch.distributed as dist


# set random seed 
def set_seed(seed, base=0, is_set=True):
  seed += base
  assert seed >=0, '{} >= {}'.format(seed, 0)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = True

class ZipReader(object):
  file_dict = dict()
  def __init__(self):
    super(ZipReader, self).__init__()

  @staticmethod
  def build_file_dict(path):
    file_dict = ZipReader.file_dict
    if path in file_dict:
      return file_dict[path]
    else:
      file_handle = zipfile.ZipFile(path, mode='r', allowZip64=True)
      file_dict[path] = file_handle
      return file_dict[path]

  @staticmethod
  def imread(path, image_name):
    zfile = ZipReader.build_file_dict(path)
    data = zfile.read(image_name)
    im = Image.open(io.BytesIO(data))
    return im

# set parameter to gpu or cpu
def set_device(args):
  if torch.cuda.is_available():
    if isinstance(args, list):
      return (item.cuda() for item in args)
    else:
      return args.cuda()
  return args

def postprocess(img):
  img = (img+1)/2*255
  img = img.permute(0,2,3,1)
  img = img.int().cpu().numpy().astype(np.uint8)
  return img


class Progbar(object):
  """Displays a progress bar.

  Arguments:
    target: Total number of steps expected, None if unknown.
    width: Progress bar width on screen.
    verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    stateful_metrics: Iterable of string names of metrics that
      should *not* be averaged over time. Metrics in this list
      will be displayed as-is. All others will be averaged
      by the progbar before display.
    interval: Minimum visual progress update interval (in seconds).
  """

  def __init__(self, target, width=25, verbose=1, interval=0.05, stateful_metrics=None):
    self.target = target
    self.width = width
    self.verbose = verbose
    self.interval = interval
    if stateful_metrics:
      self.stateful_metrics = set(stateful_metrics)
    else:
      self.stateful_metrics = set()

    self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
      sys.stdout.isatty()) or 'ipykernel' in sys.modules or 'posix' in sys.modules)
    self._total_width = 0
    self._seen_so_far = 0
    # We use a dict + list to avoid garbage collection
    # issues found in OrderedDict
    self._values = {}
    self._values_order = []
    self._start = time.time()
    self._last_update = 0

  def update(self, current, values=None):
    """Updates the progress bar.
    Arguments:
      current: Index of current step.
      values: List of tuples:
        `(name, value_for_last_step)`.
        If `name` is in `stateful_metrics`,
        `value_for_last_step` will be displayed as-is.
        Else, an average of the metric over time will be displayed.
    """
    values = values or []
    for k, v in values:
      if k not in self._values_order:
        self._values_order.append(k)
      if k not in self.stateful_metrics:
        if k not in self._values:
          self._values[k] = [v * (current - self._seen_so_far), current - self._seen_so_far]
        else:
          self._values[k][0] += v * (current - self._seen_so_far)
          self._values[k][1] += (current - self._seen_so_far)
      else:
        self._values[k] = v
    self._seen_so_far = current

    now = time.time()
    info = ' - %.0fs' % (now - self._start)
    if self.verbose == 1:
      if (now - self._last_update < self.interval and 
        self.target is not None and current < self.target):
          return

      prev_total_width = self._total_width
      if self._dynamic_display:
        sys.stdout.write('\b' * prev_total_width)
        sys.stdout.write('\r')
      else:
        sys.stdout.write('\n')

      if self.target is not None:
        numdigits = int(np.floor(np.log10(self.target))) + 1
        barstr = '%%%dd/%d [' % (numdigits, self.target)
        bar = barstr % current
        prog = float(current) / self.target
        prog_width = int(self.width * prog)
        if prog_width > 0:
          bar += ('=' * (prog_width - 1))
          if current < self.target:
            bar += '>'
          else:
            bar += '='
        bar += ('.' * (self.width - prog_width))
        bar += ']'
      else:
        bar = '%7d/Unknown' % current
      self._total_width = len(bar)
      sys.stdout.write(bar)
      if current:
        time_per_unit = (now - self._start) / current
      else:
        time_per_unit = 0
      if self.target is not None and current < self.target:
        eta = time_per_unit * (self.target - current)
        if eta > 3600:
          eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
        elif eta > 60:
          eta_format = '%d:%02d' % (eta // 60, eta % 60)
        else:
          eta_format = '%ds' % eta
        info = ' - ETA: %s' % eta_format
      else:
        if time_per_unit >= 1:
          info += ' %.0fs/step' % time_per_unit
        elif time_per_unit >= 1e-3:
          info += ' %.0fms/step' % (time_per_unit * 1e3)
        else:
          info += ' %.0fus/step' % (time_per_unit * 1e6)

      for k in self._values_order:
        info += ' - %s:' % k
        if isinstance(self._values[k], list):
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if abs(avg) > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        else:
          info += ' %s' % self._values[k]

      self._total_width += len(info)
      if prev_total_width > self._total_width:
        info += (' ' * (prev_total_width - self._total_width))
      if self.target is not None and current >= self.target:
        info += '\n'
      sys.stdout.write(info)
      sys.stdout.flush()
    elif self.verbose == 2:
      if self.target is None or current >= self.target:
        for k in self._values_order:
          info += ' - %s:' % k
          avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
          if avg > 1e-3:
            info += ' %.4f' % avg
          else:
            info += ' %.4e' % avg
        info += '\n'
        sys.stdout.write(info)
        sys.stdout.flush()
    self._last_update = now

  def add(self, n, values=None):
    self.update(self._seen_so_far + n, values)


# #####################################
# ############ painter ################
# #####################################

# Python 2/3 compatibility
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    from functools import reduce

import numpy as np
import cv2 as cv

# built-in modules
import os
import itertools as it
from contextlib import contextmanager

image_extensions = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.pbm', '.pgm', '.ppm']

class Bunch(object):
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __str__(self):
        return str(self.__dict__)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def anorm2(a):
    return (a*a).sum(-1)
def anorm(a):
    return np.sqrt( anorm2(a) )

def homotrans(H, x, y):
    xs = H[0, 0]*x + H[0, 1]*y + H[0, 2]
    ys = H[1, 0]*x + H[1, 1]*y + H[1, 2]
    s  = H[2, 0]*x + H[2, 1]*y + H[2, 2]
    return xs/s, ys/s

def to_rect(a):
    a = np.ravel(a)
    if len(a) == 2:
        a = (0, 0, a[0], a[1])
    return np.array(a, np.float64).reshape(2, 2)

def rect2rect_mtx(src, dst):
    src, dst = to_rect(src), to_rect(dst)
    cx, cy = (dst[1] - dst[0]) / (src[1] - src[0])
    tx, ty = dst[0] - src[0] * (cx, cy)
    M = np.float64([[ cx,  0, tx],
                    [  0, cy, ty],
                    [  0,  0,  1]])
    return M


def lookat(eye, target, up = (0, 0, 1)):
    fwd = np.asarray(target, np.float64) - eye
    fwd /= anorm(fwd)
    right = np.cross(fwd, up)
    right /= anorm(right)
    down = np.cross(fwd, right)
    R = np.float64([right, down, fwd])
    tvec = -np.dot(R, eye)
    return R, tvec

def mtx2rvec(R):
    w, u, vt = cv.SVDecomp(R - np.eye(3))
    p = vt[0] + u[:,0]*w[0]    # same as np.dot(R, vt[0])
    c = np.dot(vt[0], p)
    s = np.dot(vt[1], p)
    axis = np.cross(vt[0], vt[1])
    return axis * np.arctan2(s, c)

def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x+1, y+1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)

class Sketcher:
    def __init__(self, windowname, dests, colors_func, thick):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        self.thick = thick
        cv.setMouseCallback(self.windowname, self.on_mouse)

    def show(self):
        cv.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv.line(dst, self.prev_pt, pt, color, self.thick)
            self.dirty = True
            self.prev_pt = pt
            self.show()


# palette data from matplotlib/_cm.py
_jet_data =   {'red':   ((0., 0, 0), (0.35, 0, 0), (0.66, 1, 1), (0.89,1, 1),
                         (1, 0.5, 0.5)),
               'green': ((0., 0, 0), (0.125,0, 0), (0.375,1, 1), (0.64,1, 1),
                         (0.91,0,0), (1, 0, 0)),
               'blue':  ((0., 0.5, 0.5), (0.11, 1, 1), (0.34, 1, 1), (0.65,0, 0),
                         (1, 0, 0))}

cmap_data = { 'jet' : _jet_data }

def make_cmap(name, n=256):
    data = cmap_data[name]
    xs = np.linspace(0.0, 1.0, n)
    channels = []
    eps = 1e-6
    for ch_name in ['blue', 'green', 'red']:
        ch_data = data[ch_name]
        xp, yp = [], []
        for x, y1, y2 in ch_data:
            xp += [x, x+eps]
            yp += [y1, y2]
        ch = np.interp(xs, xp, yp)
        channels.append(ch)
    return np.uint8(np.array(channels).T*255)

def nothing(*arg, **kw):
    pass

def clock():
    return cv.getTickCount() / cv.getTickFrequency()

@contextmanager
def Timer(msg):
    print(msg, '...',)
    start = clock()
    try:
        yield
    finally:
        print("%.2f ms" % ((clock()-start)*1000))

class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v

class RectSelector:
    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None
    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return
        if self.drag_start:
            if flags & cv.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)
    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True
    @property
    def dragging(self):
        return self.drag_rect is not None


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    if PY3:
        output = it.zip_longest(fillvalue=fillvalue, *args)
    else:
        output = it.izip_longest(fillvalue=fillvalue, *args)
    return output

def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    if PY3:
        img0 = next(imgs)
    else:
        img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))

def getsize(img):
    h, w = img.shape[:2]
    return w, h

def mdot(*args):
    return reduce(np.dot, args)

def draw_keypoints(vis, keypoints, color = (0, 255, 255)):
    for kp in keypoints:
        x, y = kp.pt
        cv.circle(vis, (int(x), int(y)), 2, color)

# ####################################################################
# ###################################################################
from typing import Tuple

import numpy as np 
from PIL import Image 
import torchvision.transforms.functional as F
import cv2

import torch
import torch.nn as nn
from torch.nn.functional import conv2d


def gaussian(window_size, sigma):
  def gauss_fcn(x):
    return -(x - window_size // 2)**2 / float(2 * sigma**2)
  gauss = torch.stack([torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
  return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
  r"""Function that returns Gaussian filter coefficients.
  Args:
    kernel_size (int): filter size. It should be odd and positive.
    sigma (float): gaussian standard deviation.
  Returns:
    Tensor: 1D tensor with gaussian filter coefficients.
  Shape:
    - Output: :math:`(\text{kernel_size})`

  Examples::
    >>> kornia.image.get_gaussian_kernel(3, 2.5)
    tensor([0.3243, 0.3513, 0.3243])
    >>> kornia.image.get_gaussian_kernel(5, 1.5)
    tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
  """
  if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or kernel_size <= 0:
    raise TypeError("kernel_size must be an odd positive integer. Got {}".format(kernel_size))
  window_1d: torch.Tensor = gaussian(kernel_size, sigma)
  return window_1d


def get_gaussian_kernel2d(kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> torch.Tensor:
  r"""Function that returns Gaussian filter matrix coefficients.
  Args:
    kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
      Sizes should be odd and positive.
    sigma (Tuple[int, int]): gaussian standard deviation in the x and y
      direction.
  Returns:
    Tensor: 2D tensor with gaussian filter matrix coefficients.

  Shape:
    - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

  Examples::
    >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
    tensor([[0.0947, 0.1183, 0.0947],
            [0.1183, 0.1478, 0.1183],
            [0.0947, 0.1183, 0.0947]])

    >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
    tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
            [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
            [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
  """
  if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
    raise TypeError("kernel_size must be a tuple of length two. Got {}".format(kernel_size))
  if not isinstance(sigma, tuple) or len(sigma) != 2:
    raise TypeError("sigma must be a tuple of length two. Got {}".format(sigma))
  ksize_x, ksize_y = kernel_size
  sigma_x, sigma_y = sigma
  kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
  kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
  kernel_2d: torch.Tensor = torch.matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
  return kernel_2d

class GaussianBlur(nn.Module):
  r"""Creates an operator that blurs a tensor using a Gaussian filter.
  The operator smooths the given tensor with a gaussian kernel by convolving
  it to each channel. It suports batched operation.
  Arguments:
    kernel_size (Tuple[int, int]): the size of the kernel.
    sigma (Tuple[float, float]): the standard deviation of the kernel.
  Returns:
    Tensor: the blurred tensor.
  Shape:
    - Input: :math:`(B, C, H, W)`
    - Output: :math:`(B, C, H, W)`

  Examples::
    >>> input = torch.rand(2, 4, 5, 5)
    >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
    >>> output = gauss(input)  # 2x4x5x5
  """

  def __init__(self, kernel_size: Tuple[int, int], sigma: Tuple[float, float]) -> None:
    super(GaussianBlur, self).__init__()
    self.kernel_size: Tuple[int, int] = kernel_size
    self.sigma: Tuple[float, float] = sigma
    self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
    self.kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)

  @staticmethod
  def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
    """Computes zero padding tuple."""
    computed = [(k - 1) // 2 for k in kernel_size]
    return computed[0], computed[1]

  def forward(self, x: torch.Tensor):  # type: ignore
    if not torch.is_tensor(x):
      raise TypeError("Input x type is not a torch.Tensor. Got {}".format(type(x)))
    if not len(x.shape) == 4:
      raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}".format(x.shape))
    # prepare kernel
    b, c, h, w = x.shape
    tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
    kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

    # TODO: explore solution when using jit.trace since it raises a warning
    # because the shape is converted to a tensor instead to a int.
    # convolve tensor with gaussian kernel
    return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)



######################
# functional interface
######################

def gaussian_blur(input: torch.Tensor, kernel_size: Tuple[int, int],
  sigma: Tuple[float, float]) -> torch.Tensor:
  r"""Function that blurs a tensor using a Gaussian filter.
  See :class:`~kornia.filters.GaussianBlur` for details.
  """
  return GaussianBlur(kernel_size, sigma)(input)

if __name__ == '__main__':
  img = Image.open('test.png').convert('L')
  tensor_img = F.to_tensor(img).unsqueeze(0).float()
  print('tensor_img size: ', tensor_img.size())

  blurred_img = gaussian_blur(tensor_img, (61,61), (10, 10))
  print(torch.min(blurred_img), torch.max(blurred_img))


  blurred_img = blurred_img*255
  img = blurred_img.int().numpy().astype(np.uint8)[0][0]
  print(img.shape, np.min(img), np.max(img), np.unique(img))
  cv2.imwrite('gaussian.png', img)

