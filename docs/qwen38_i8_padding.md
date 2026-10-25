# Qwen3.8 INT8 gate/up M24 padding

本改动只保留一个已验证的GEMM分派：`(M,N,K)=(24,34816,5120)`。
`M` 是实际token行数，不是请求并发数；9～15并发工作负载命中该尺寸时可使用此路径，
其他尺寸保留原生实现。

汇编核将输入从24行补齐到32行，保留有效输入和行scale，然后调用原生hipBLASLt，
最后拷贝前24行BF16结果。原始INT8权重布局、INT32累加、FP32缩放及BF16输出不变。
这不是完整GEMM汇编替换，不包含vLLM调度、输出头量化和其他实验。

## 文件布局

| 文件 | 作用 |
|---|---|
| `python/lightop/qwen38_i8_padding.py` | 补齐分派和原生GEMM调用 |
| `python/lightop/_lmslim_native/quantize/quant_ops.py` | 现有W8A8 BF16接口中的可选分派入口 |
| `csrc/asm_cu/qwen38_i8_pad16.s` | gfx936输入及行scale补齐汇编 |
| `hsa/gfx936/qwen38_i8_pad16.co` | 已验证的code object |

`pad16` 是沿用的符号名；实际补齐行数由启动网格指定，本分派使用32行。

## 启用

按仓库现有方式构建、安装LightOp后，在启动Python进程之前设置：

```bash
export LIGHTOP_QWEN38_PAD_ROUTE=1
```

默认关闭。不匹配的尺寸、dtype、布局、转置或bias保留原生调用。
硬件限定gfx936，验证环境为当前DTK镜像。首次调用需发生在Graph捕获之前；
直接使用Graph的调用方应在同一进程先调用
`lightop.qwen38_i8_padding.initialize()`，或进行一次常规预热。

默认读取安装包内的`hsa/gfx936/qwen38_i8_pad16.co`，也可通过
`LIGHTOP_QWEN38_PAD_HSA`指定产物路径。

## 重建补齐核

在仓库根目录执行，先另存产物以保留已验证的二进制：

```bash
/opt/dtk/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx936 \
  -mcode-object-version=4 -c csrc/asm_cu/qwen38_i8_pad16.s \
  -o /tmp/qwen38_i8_pad16.o
/opt/dtk/llvm/bin/ld.lld -shared /tmp/qwen38_i8_pad16.o \
  -o /tmp/qwen38_i8_pad16.rebuilt.co
```

## 已有验证范围

完整有符号INT8随机操作数检查中，补齐路径的BF16输出与原生结果逐位一致。
既有合成微测包含补齐、GEMM和有效输出拷贝，四组独立权重轮换、Graph内计时及
三轮交替对照：裸M24约221µs，补齐链约172µs，约1.28倍。

该结果是单算子微测，没有证明整个9～15并发范围的独立端到端收益；此前包含调度
改动的整机结果不作为此分派的独立收益，也不宣称已达到TPOT≤20ms。
