# SPDX-License-Identifier: Apache-2.0
"""Optional exact M24 gate/up W8A8 route with bounded activation workspace.

The original INT8 weight layout is retained. A small gfx936 ISA kernel pads A and
its per-token scale to 32 rows; native INT8 GEMM then avoids a slow small-M tile.
Only (M,N,K)=(24,34816,5120) is selected. No precision change.
"""
import ctypes as C
import os
from pathlib import Path
import threading
import torch

ENABLED=os.getenv('LIGHTOP_QWEN38_PAD_ROUTE','0')=='1'
_LOCK=threading.Lock()
_KERNELS={}
_HIP=None
_ROUTES={(24,34816,5120):32}

class _Args(C.Structure):
    _fields_=[(name,C.c_uint64) for name in ('a','sa','apad','sapad')]+[(name,C.c_uint32) for name in ('m','k')]

def _check(status):
    if status:raise RuntimeError('LightOp PAD_ROUTE: '+_HIP.hipGetErrorString(status).decode())

def initialize(device=None):
    """Resolve the module outside graph capture, once per device."""
    global _HIP
    if device is None:device=torch.cuda.current_device()
    if device in _KERNELS:return True
    with _LOCK,torch.cuda.device(device):
        if device in _KERNELS:return True
        if getattr(torch.cuda.get_device_properties(device),'gcnArchName','').split(':')[0]!='gfx936':return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Initialize LightOp PAD_ROUTE before graph capture')
        if _HIP is None:
            _HIP=C.CDLL(os.getenv('LIGHTOP_QWEN38_HIP_LIBRARY','/opt/dtk/lib/libamdhip64.so'))
            pointer=C.c_void_p
            _HIP.hipModuleLoad.argtypes=[C.POINTER(pointer),C.c_char_p]
            _HIP.hipModuleGetFunction.argtypes=[C.POINTER(pointer),pointer,C.c_char_p]
            _HIP.hipModuleLaunchKernel.argtypes=[pointer,*([C.c_uint]*7),pointer,pointer,C.POINTER(pointer)]
            _HIP.hipGetErrorString.argtypes=[C.c_int];_HIP.hipGetErrorString.restype=C.c_char_p
        path=Path(os.getenv('LIGHTOP_QWEN38_PAD_HSA',str(Path(__file__).parent/'hsa/gfx936/qwen38_i8_pad16.co')))
        if not path.is_file():raise FileNotFoundError(path)
        module=C.c_void_p();_check(_HIP.hipModuleLoad(C.byref(module),str(path).encode()))
        function=C.c_void_p();_check(_HIP.hipModuleGetFunction(C.byref(function),module,b'qwen38_i8_pad16'))
        _KERNELS[device]=(module,function)
    return True

def maybe_run(a,weight,sa,sw,output,m,n,k,batch,transpose,bias,native,alpha,beta):
    """Return True only after writing output; otherwise retain native behavior."""
    if not ENABLED or (m,n,k) not in _ROUTES or batch!=1 or transpose!='TN' or bias is not None:
        return False
    tensors=(a,weight,sa,sw,output)
    if not (a.is_cuda and all(t.device==a.device and t.is_contiguous() for t in tensors)
            and a.dtype==weight.dtype==torch.int8 and sa.dtype==sw.dtype==torch.float32 and output.dtype==torch.bfloat16
            and tuple(a.shape)==(m,k) and tuple(weight.shape)==(n,k) and tuple(output.shape)==(m,n)
            and sa.numel()==m and sw.numel()==n):return False
    if not (alpha.device.type==beta.device.type=='cpu' and alpha.numel()==beta.numel()==1
            and alpha.item()==1 and beta.item()==0):return False
    padded_m=_ROUTES[(m,n,k)]
    with torch.cuda.device(a.device):
        if a.device.index not in _KERNELS and not initialize(a.device.index):return False
        apad=torch.empty((padded_m,k),device=a.device,dtype=torch.int8)
        sapad=torch.empty((padded_m,1),device=a.device,dtype=torch.float32)
        padded=torch.empty((padded_m,n),device=a.device,dtype=torch.bfloat16)
        arguments=_Args(*(tensor.data_ptr() for tensor in (a,sa,apad,sapad)),m,k);size=C.c_size_t(40)
        pointer=C.c_void_p;config=(pointer*5)(1,C.addressof(arguments),2,C.addressof(size),3)
        function=_KERNELS[a.device.index][1]
        _check(_HIP.hipModuleLaunchKernel(function,padded_m,(k+4095)//4096,1,256,1,1,0,
            pointer(torch.cuda.current_stream(a.device).cuda_stream),None,config))
        native(weight,apad,sw,sapad,padded,padded_m,n,k,batch,transpose,alpha,beta,None)
        output.copy_(padded[:m])
    return True
