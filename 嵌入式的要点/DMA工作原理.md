# DMA 工作原理：CPU、外设与 DMA 如何配合

## 核心理解

DMA（Direct Memory Access）的作用，是在外设和内存之间搬运数据时，避免 CPU 逐字节参与。

CPU 主要负责：**提前配置并启动 DMA，以及在传输完成后处理结果**。真正的数据搬运主要由 DMA 控制器完成。

## UART + DMA 接收的典型流程

以 UART 使用 DMA 接收 100 字节为例：

```text
CPU：配置 DMA
    - 源地址：UART 数据寄存器
    - 目标地址：RAM buffer
    - 长度：100 字节
    - 方向：外设 -> 内存
        ↓
CPU：使能 DMA
        ↓
DMA 已经“武装好”，等待 UART 的 DMA Request
        ↓
UART 收到第 1 字节 -> UART 向 DMA 发请求 -> DMA 搬到 buffer[0]
UART 收到第 2 字节 -> UART 向 DMA 发请求 -> DMA 搬到 buffer[1]
...
UART 收到第 100 字节 -> DMA 搬到 buffer[99]
        ↓
DMA 传输完成
        ↓
DMA 产生完成中断通知 CPU
        ↓
CPU 处理这一批数据，并准备下一轮 DMA
```

## CPU 是什么时候启动 DMA 的？

重点是：**CPU 不是等数据来了以后才启动 DMA，而是提前启动。**

第一次通常在系统初始化阶段：

```text
初始化时钟
  ↓
初始化 GPIO
  ↓
初始化 UART
  ↓
初始化 DMA
  ↓
启动 DMA 接收
  ↓
进入主循环 / 启动 RTOS
```

这时 UART 可能一个字节都还没有收到，DMA 已经在等待数据。

普通 DMA 模式下，一轮传输完成后，DMA 可以中断 CPU，CPU 再重新设置长度、缓冲区等并使能 DMA，从而**提前准备下一轮数据**。

## 两种“通知”不要混淆

### 外设 -> DMA：DMA Request

UART 每收到数据后，可以直接通过硬件向 DMA 发请求。这个请求不是 CPU 中断，因此不需要 CPU 先知道“来了一个字节”。

### DMA -> CPU：Interrupt

DMA 达到某个条件后才通知 CPU，例如：

- Half Transfer：传输一半
- Transfer Complete：传输完成
- Transfer Error：传输出错

因此典型关系是：

```text
外设 <-> DMA <-> RAM
          |
          | 完成/异常
          v
         CPU
```

## DMA 工作期间 CPU 能做什么？

DMA 搬运数据期间 CPU 可以继续：

- 执行主程序
- 运行 RTOS 任务
- 处理中断
- 执行计算

CPU 进入其他中断时，DMA 通常仍然能够继续工作。CPU 与 DMA 可能因为同时访问总线、RAM 等资源产生短暂的总线仲裁，但 CPU 不需要逐字节参与 DMA 搬运。

## 普通 DMA 与循环 DMA

普通模式通常是：

```text
CPU 启动 DMA
 -> DMA 完成一轮
 -> 中断 CPU
 -> CPU 准备下一轮
 -> DMA 再等待下一批数据
```

循环 DMA（Circular DMA）则可以在一次启动后自动循环使用缓冲区：

```text
CPU 启动一次
 -> DMA 填满 buffer
 -> 自动回到 buffer 开头
 -> 继续接收
 -> 自动循环……
```

因此循环 DMA 可以进一步减少 CPU 对 DMA 的重新启动操作。

## 一句话总结

> **CPU 提前“武装” DMA；数据到来后，外设直接请求 DMA 搬运；DMA 完成一批数据后，再通过中断通知 CPU。**

所以并不是：

```text
数据来了 -> 中断 CPU -> CPU 启动 DMA -> DMA 搬数据
```

而通常是：

```text
CPU 提前启动 DMA
       ↓
DMA 等待
       ↓
数据到来
       ↓
外设 DMA Request
       ↓
DMA 自动搬运
       ↓
传输完成
       ↓
中断 CPU
```
