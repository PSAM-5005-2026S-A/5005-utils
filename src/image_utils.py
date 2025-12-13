import numpy as np

from PIL import Image as PImage, ImageFilter as PImageFilter

## Image Helpers

def to1d(ndarr):
  if ndarr.shape[-1] == 3 or ndarr.shape[-1] == 4:
    return ndarr.reshape(-1, ndarr.shape[-1]).tolist()
  return ndarr.reshape(-1).tolist()


def get_pixels(img):
  return list(img.getdata())


def make_image(pxs, width=None, height=None):
  if hasattr(pxs, "iloc") and hasattr(pxs, "values"):
    pxs = pxs.values
    if len(pxs.shape) < 2:
      pxs = pxs.reshape(-1, 1)
    pxs = to1d(pxs)
  elif hasattr(pxs, "shape") and hasattr(pxs, "reshape"):
    if hasattr(pxs, "int"):
      pxs = pxs.int()
    elif hasattr(pxs, "astype"):
      pxs = pxs.astype("int")
    pxs = to1d(pxs)

  MODES = ["", "L", "XX", "RGB", "RGBA"]
  nw = int(len(pxs) ** 0.5) if width is None else width
  nh = int(len(pxs) // nw) if height is None else height

  pxs = [tuple(p) for p in pxs] if type(pxs[0]) is list else pxs

  nc = len(pxs[0]) if type(pxs[0]) is tuple else 1

  mimg = PImage.new(MODES[nc], (nw,nh))
  mimg.putdata(pxs[ :(nw * nh)])

  return mimg



## Edge Detection
def constrain_uint8(v):
  return int(min(max(v, 0), 255))

def blur(img, rad=1.0):
  return img.filter(PImageFilter.GaussianBlur(rad))

def edges_rgb(img, rad=1.0):
  bimg = blur(img, rad)
  pxs = get_pixels(img)
  bpxs = get_pixels(bimg)

  bdiffpx = []
  for (r0,g0,b0), (r1,g1,b1) in zip(bpxs, pxs):
    bdiffpx.append((
      constrain_uint8(np.exp(r1-r0)),
      constrain_uint8(np.exp(g1-g0)),
      constrain_uint8(np.exp(b1-b0)),
    ))

  bimg = make_image(bdiffpx, bimg.size[0])
  return bimg

def edges_exp_thold(img, rad=1.0):
  bimg = blur(img, rad)
  pxs = get_pixels(img)
  bpxs = get_pixels(bimg)

  bdiffpx = []
  for (r0,g0,b0), (r1,g1,b1) in zip(bpxs, pxs):
    bdiffpx.append(constrain_uint8(np.exp(r1-r0)))

  bimg = make_image(bdiffpx, bimg.size[0])
  return bimg

def edges(img, rad=1, thold=16):
  bimg = blur(img, rad)

  # get luminance
  gipxs = [(r+g+b)//3 for r,g,b in get_pixels(img)]
  gbpxs = [(r+g+b)//3 for r,g,b in get_pixels(bimg)]

  # subtract and threshold
  epxs = [255 if (o-b)>thold else 0 for o,b in zip(gipxs, gbpxs)]

  # prepare output
  eimg = make_image(epxs, img.size[0])
  return eimg
