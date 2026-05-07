import argparse
import features
import math
import model as M
import numpy
import struct
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from functools import reduce
import operator

def ascii_hist(name, x, bins=6):
  N,X = numpy.histogram(x, bins=bins)
  total = 1.0*len(x)
  width = 50
  nmax = N.max()

  print(name)
  for (xi, n) in zip(X,N):
    bar = '#'*int(n*1.0*width/nmax)
    xi = '{0: <8.4g}'.format(xi).ljust(10)
    print('{0}| {1}'.format(xi,bar))

def feature_set_for_checkpoint(source, requested_feature_set):
  checkpoint = torch.load(source, map_location='cpu')
  state_dict = checkpoint.get('state_dict', checkpoint)
  input_weight = state_dict.get('input.weight')

  if input_weight is None:
    return requested_feature_set

  checkpoint_features = input_weight.shape[0]
  if checkpoint_features == requested_feature_set.num_features:
    return requested_feature_set

  if checkpoint_features == requested_feature_set.num_real_features:
    real_feature_name = '+'.join(feature.get_main_factor_name() for feature in requested_feature_set.features)
    real_feature_set = features.get_feature_set_from_name(real_feature_name)
    if checkpoint_features == real_feature_set.num_features:
      print('Checkpoint input has {} features; loading with real feature set {} instead of requested training feature set {}.'.format(
        checkpoint_features, real_feature_set.name, requested_feature_set.name))
      return real_feature_set

  raise RuntimeError('Checkpoint input.weight has {} features, but requested feature set {} has {} training features and {} real features.'.format(
    checkpoint_features, requested_feature_set.name, requested_feature_set.num_features, requested_feature_set.num_real_features))

VERSION = 0x7AF32F20
DEFAULT_DESCRIPTION = "Network trained with the https://github.com/ianfab/variant-nnue-pytorch trainer."

class NNUEWriter():
  def __init__(self, model, description=None):
    if description is None:
        description = DEFAULT_DESCRIPTION

    self.buf = bytearray()

    fc_hash = self.fc_hash(model)
    self.write_header(model, fc_hash, description)
    self.int32(model.feature_set.hash ^ (model.l1_size*2))
    self.write_feature_transformer(model)
    for l1, l2, output in model.layer_stacks.get_coalesced_layer_stacks():
      self.int32(fc_hash)
      self.write_fc_layer(l1)
      self.write_fc_layer(l2)
      self.write_fc_layer(output, is_output=True)

  @staticmethod
  def fc_hash(model):
    prev_hash = 0xEC42E90D
    prev_hash ^= (model.l1_size * 2)
    layers = [model.layer_stacks.l1, model.layer_stacks.l2, model.layer_stacks.output]
    for layer in layers:
      layer_hash = 0xCC03DAE4
      layer_hash += layer.out_features // model.num_ls_buckets
      layer_hash ^= prev_hash >> 1
      layer_hash ^= (prev_hash << 31) & 0xFFFFFFFF
      if layer.out_features // model.num_ls_buckets != 1:
        layer_hash = (layer_hash + 0x538D24C7) & 0xFFFFFFFF
      prev_hash = layer_hash
    return layer_hash

  def write_header(self, model, fc_hash, description):
    self.int32(VERSION)
    self.int32(fc_hash ^ model.feature_set.hash ^ (model.l1_size*2))
    self.int32(model.l1_size)
    self.int32(model.l2_size)
    encoded_description = description.encode('utf-8')
    self.int32(len(encoded_description))
    self.buf.extend(encoded_description)

  def write_feature_transformer(self, model):
    layer = model.input
    bias = layer.bias.data[:model.l1_size]
    bias = bias.mul(127).round().to(torch.int16)
    self.buf.extend(bias.flatten().numpy().tobytes())

    weight = M.coalesce_ft_weights(model, layer)
    weight0 = weight[:, :model.l1_size]
    psqtweight0 = weight[:, model.l1_size:]
    weight = weight0.mul(127).round().to(torch.int16)
    psqtweight = psqtweight0.mul(9600).round().to(torch.int32)
    self.buf.extend(weight.flatten().numpy().tobytes())
    self.buf.extend(psqtweight.flatten().numpy().tobytes())

  def write_fc_layer(self, layer, is_output=False):
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale
    else:
      kBiasScale = 9600.0
    kWeightScale = kBiasScale / kActivationScale
    kMaxWeight = 127.0 / kWeightScale

    bias = layer.bias.data.mul(kBiasScale).round().to(torch.int32)
    self.buf.extend(bias.flatten().numpy().tobytes())
    weight = layer.weight.data.clamp(-kMaxWeight, kMaxWeight).mul(kWeightScale).round().to(torch.int8)

    num_input = weight.shape[1]
    if num_input % 32 != 0:
      num_input += 32 - (num_input % 32)
      new_w = torch.zeros(weight.shape[0], num_input, dtype=torch.int8)
      new_w[:, :weight.shape[1]] = weight
      weight = new_w
    self.buf.extend(weight.flatten().numpy().tobytes())

  def int32(self, v):
    self.buf.extend(struct.pack("<I", v))


class NNUEReader():
  def __init__(self, f, feature_set, forced_l1=None, forced_l2=None):
    self.f = f
    self.feature_set = feature_set
    self.model = None

    l1_size, l2_size, header_kind = self.read_header(feature_set, forced_l1=forced_l1, forced_l2=forced_l2)
    print(f"NNUEReader: using header mode={header_kind}, l1={l1_size}, l2={l2_size}")
    self.model = M.NNUE(feature_set, l1_size=l1_size, l2_size=l2_size)
    fc_hash = NNUEWriter.fc_hash(self.model)

    if header_kind == 'trainer':
      self.read_int32(feature_set.hash ^ (self.model.l1_size * 2), strict=False, label='feature transformer hash')
    else:
      self.read_int32(label='feature transformer hash (legacy)')

    self.read_feature_transformer(self.model.input, self.model.num_psqt_buckets)

    for i in range(self.model.num_ls_buckets):
      l1 = nn.Linear(2 * self.model.l1_size, self.model.l2_size)
      l2 = nn.Linear(self.model.l2_size, M.L3)
      output = nn.Linear(M.L3, 1)
      self.read_int32(fc_hash, strict=False, label='fc hash')
      self.read_fc_layer(l1)
      self.read_fc_layer(l2)
      self.read_fc_layer(output, is_output=True)
      self.model.layer_stacks.l1.weight.data[i*self.model.l2_size:(i+1)*self.model.l2_size, :] = l1.weight
      self.model.layer_stacks.l1.bias.data[i*self.model.l2_size:(i+1)*self.model.l2_size] = l1.bias
      self.model.layer_stacks.l2.weight.data[i*M.L3:(i+1)*M.L3, :] = l2.weight
      self.model.layer_stacks.l2.bias.data[i*M.L3:(i+1)*M.L3] = l2.bias
      self.model.layer_stacks.output.weight.data[i:(i+1), :] = output.weight
      self.model.layer_stacks.output.bias.data[i:(i+1)] = output.bias

  def read_header(self, feature_set, forced_l1=None, forced_l2=None):
    start = self.f.tell()
    self.read_int32(VERSION)
    net_hash = self.read_int32(label='network hash')

    l1_candidate = self.read_int32(label='l1 candidate')
    l2_candidate = self.read_int32(label='l2 candidate')
    trainer_like = (16 <= l1_candidate <= 65536) and (1 <= l2_candidate <= 4096)

    if trainer_like:
      desc_len = self.read_int32(label='description length')
      if 0 <= desc_len <= 10_000_000:
        _ = self.f.read(desc_len)
        if forced_l1 is not None and forced_l2 is not None:
          print(f"NNUEReader: forcing dimensions from CLI: l1={forced_l1}, l2={forced_l2}")
          return forced_l1, forced_l2, 'forced'
        temp_model = M.NNUE(feature_set, l1_size=l1_candidate, l2_size=l2_candidate)
        fc_hash = NNUEWriter.fc_hash(temp_model)
        expected = fc_hash ^ feature_set.hash ^ (l1_candidate * 2)
        if net_hash != expected:
          print(f"NNUEReader: warning header hash mismatch (expected {expected:08x}, got {net_hash:08x}); continuing in trainer mode.")
        return l1_candidate, l2_candidate, 'trainer'

    # Fallback: legacy/original C++ format where feature-transformer hash starts right after net hash.
    self.f.seek(start + 8)
    print('NNUEReader: legacy/original C++ header detected (no explicit l1/l2 header fields).')
    if forced_l1 is not None and forced_l2 is not None:
      print(f"NNUEReader: forcing dimensions from CLI: l1={forced_l1}, l2={forced_l2}")
      return forced_l1, forced_l2, 'forced'
    return M.DEFAULT_L1, M.DEFAULT_L2, 'legacy'

  def tensor(self, dtype, shape, label='tensor'):
    expected = reduce(operator.mul, shape, 1)
    d = numpy.fromfile(self.f, dtype, expected)
    got = d.size
    if got != expected:
      bytes_per_elem = numpy.dtype(dtype).itemsize
      print(f"ERROR while reading {label}: got {got} elements, expected {expected} ({expected * bytes_per_elem} bytes).")
      remaining_bytes = 0
      try:
        cur = self.f.tell()
        self.f.seek(0, 2)
        end = self.f.tell()
        self.f.seek(cur)
        remaining_bytes = end - cur
      except Exception:
        pass
      if remaining_bytes > 0:
        hint = remaining_bytes // max(1, (self.feature_set.num_features * 2))
        print(f"Hint: remaining bytes={remaining_bytes}, rough L1 hint (int16 FT weights only)≈{hint}.")
      raise RuntimeError(f"Short read for {label}")

    try:
      d = torch.from_numpy(d.astype(numpy.float32)).reshape(shape)
    except RuntimeError as e:
      print(f"Reshape error for {label}: got elements={got}, expected shape={shape} ({expected} elements)")
      if len(shape) >= 2:
        rows = shape[0]
        approx_l1 = got // max(rows, 1)
        print(f"Hint: inferred second dimension from element count is approximately {approx_l1}.")
      raise
    return d

  def read_feature_transformer(self, layer, num_psqt_buckets):
    ft_out = layer.bias.shape[0] - num_psqt_buckets
    input_dims = layer.weight.shape[0]

    bias = self.tensor(numpy.int16, [ft_out], label='ft bias').divide(127.0)
    layer.bias.data = torch.cat([bias, torch.tensor([0] * num_psqt_buckets)])

    weights = self.tensor(numpy.int16, [input_dims, ft_out], label='ft weights').divide(127.0)
    psqtweights = self.tensor(numpy.int32, [input_dims, num_psqt_buckets], label='ft psqt').divide(9600.0)
    layer.weight.data = torch.cat([weights, psqtweights], dim=1)

  def read_fc_layer(self, layer, is_output=False):
    kWeightScaleBits = 6
    kActivationScale = 127.0
    if not is_output:
      kBiasScale = (1 << kWeightScaleBits) * kActivationScale
    else:
      kBiasScale = 9600.0
    kWeightScale = kBiasScale / kActivationScale

    non_padded_shape = layer.weight.shape
    padded_shape = (non_padded_shape[0], ((non_padded_shape[1]+31)//32)*32)

    layer.bias.data = self.tensor(numpy.int32, layer.bias.shape, label='fc bias').divide(kBiasScale)
    layer.weight.data = self.tensor(numpy.int8, padded_shape, label='fc weight').divide(kWeightScale)
    layer.weight.data = layer.weight.data[:non_padded_shape[0], :non_padded_shape[1]]

  def read_int32(self, expected=None, strict=True, label='int32'):
    raw = self.f.read(4)
    if len(raw) != 4:
      raise RuntimeError(f"Unexpected EOF while reading {label}")
    v = struct.unpack("<I", raw)[0]
    if expected is not None and v != expected:
      msg = f"Expected {label}: {expected:08x}, got {v:08x}"
      if strict:
        raise Exception(msg)
      print('NNUEReader warning:', msg)
    return v

def main():
  parser = argparse.ArgumentParser(description="Converts files between ckpt and nnue format.")
  parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
  parser.add_argument("target", help="Target file (can be .pt or .nnue)")
  parser.add_argument("--description", default=None, type=str, dest='description', help="The description string to include in the network. Only works when serializing into a .nnue file.")
  parser.add_argument("--l1-size", default=M.DEFAULT_L1, type=int, dest='l1_size', help="Hidden size override when loading .nnue/.ckpt.")
  parser.add_argument("--l2-size", default=M.DEFAULT_L2, type=int, dest='l2_size', help="Hidden size override when loading .nnue/.ckpt.")
  parser.add_argument("--force-header-sizes", action='store_true', dest='force_header_sizes', help="Force parser to use --l1-size/--l2-size for .nnue, ignoring in-file values.")
  features.add_argparse_args(parser)
  args = parser.parse_args()

  feature_set = features.get_feature_set_from_name(args.features)

  print('Converting %s to %s' % (args.source, args.target))

  if args.source.endswith('.ckpt'):
    checkpoint_feature_set = feature_set_for_checkpoint(args.source, feature_set)
    nnue = M.NNUE.load_from_checkpoint(args.source, feature_set=checkpoint_feature_set, l1_size=args.l1_size, l2_size=args.l2_size)
    nnue.eval()
  elif args.source.endswith('.pt'):
    nnue = torch.load(args.source, weights_only=False)
  elif args.source.endswith('.nnue'):
    with open(args.source, 'rb') as f:
      forced_l1 = args.l1_size if args.force_header_sizes else None
      forced_l2 = args.l2_size if args.force_header_sizes else None
      reader = NNUEReader(f, feature_set, forced_l1=forced_l1, forced_l2=forced_l2)
      nnue = reader.model
  else:
    raise Exception('Invalid network input format.')

  if args.target.endswith('.ckpt'):
    raise Exception('Cannot convert into .ckpt')
  elif args.target.endswith('.pt'):
    torch.save(nnue, args.target)
  elif args.target.endswith('.nnue'):
    writer = NNUEWriter(nnue, args.description)
    with open(args.target, 'wb') as f:
      f.write(writer.buf)
  else:
    raise Exception('Invalid network output format.')

if __name__ == '__main__':
  main()
