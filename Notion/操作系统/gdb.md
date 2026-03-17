---

---

# GDB 速记

## 编译时打开调试信息

```bash
g++ main.cpp -g -O0 -o app
```

## 启动

```bash
gdb ./app
```

## 常用命令

- `run`
  - 运行程序
- `start`
  - 停在 `main`
- `break main`
  - 在 `main` 处打断点
- `break file.cpp:20`
  - 在指定文件行打断点
- `next`
  - 单步跳过
- `step`
  - 单步进入
- `continue`
  - 继续执行
- `print x`
  - 查看变量
- `bt`
  - 查看调用栈
- `frame n`
  - 切到第 `n` 层栈帧
- `info locals`
  - 查看局部变量
- `info threads`
  - 查看线程列表
- `thread n`
  - 切换到指定线程
- `info inferiors`
  - 查看当前调试的进程列表
- `inferior n`
  - 切换到指定进程
- `quit`
  - 退出

## 条件断点

```gdb
break foo.cpp:42 if x == 10
```

- 只有条件满足时才会命中
- 适合循环很多次、只想抓特定状态的问题

## 多进程调试

```gdb
set follow-fork-mode child
set detach-on-fork off
info inferiors
inferior 2
```

- `follow-fork-mode parent|child`
  - 跟踪父进程或子进程
- `detach-on-fork off`
  - `fork` 后父子进程都保留在 GDB 控制下

## 多线程调试

```gdb
info threads
thread 3
thread apply all bt
```

- `info threads`
  - 查看线程列表
- `thread n`
  - 切换线程
- `thread apply all bt`
  - 查看所有线程调用栈

## core dump 调试

先确保程序带调试信息并允许生成 core：

```bash
g++ main.cpp -g -O0 -o app
ulimit -c unlimited
```

调试 core：

```bash
gdb ./app core
```

进入后常用：

- `bt`
- `frame n`
- `info locals`
- `print var`
- `thread apply all bt`

## 什么是 core dump

- 程序崩溃时保存的一份现场快照
- 常包含寄存器、线程栈、部分内存映像等信息
- 常用于定位段错误、非法访问、栈溢出等问题

## 传参

```gdb
set args 10 20 30
run
```

## 面试简答

- `-g` 用于加入调试符号
- `GDB` 常用于断点、单步、查看变量、查看调用栈、排查崩溃
- 条件断点适合只在满足条件时停下
- 多线程常看 `info threads` / `thread apply all bt`
- core dump 常配合 `gdb 程序 core文件` 定位崩溃现场
