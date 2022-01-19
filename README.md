# DietGPU: a lossless GPU compression library

Author: Jeff Johnson (jhj@fb.com)

DietGPU is a library for specialized lossless compression of data on the GPU.

It currently consists of two parts:

- a general range-based asymmetric numeral systems (rANS) entropy encoder and decoder, that operates at throughputs around 250-410 GB/s on an A100 GPU
- an extension to the above to handle fast lossless compression and decompression of unstructured floating point data, for use in ML and HPC applications, especially in communicating over local interconnects (PCIe / NVLink) and remote interconnects (Ethernet / InfiniBand)

The rANS codec operates on 8 bit bytes. The floating point compressor at the moment uses the rANS codec to handle compression of floating point exponents, as typically in ML/HPC data a very limited exponent dynamic range is used and is highly compressible. Floating point sign and significand values tend to be less compressible / fairly high entropy in practice, though sparse data or presence of functions like ReLU in neural networks can result in a lot of outright zero values which are very compressible. A future extension to the library will allow for specialized compression of sparse or semi-sparse data, specializing compression of zeros.

Both APIs are available in both C++ (raw pointers) and Python/PyTorch (PyTorch tensor) API forms.
