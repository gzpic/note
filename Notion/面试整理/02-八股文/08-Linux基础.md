# Linux 基础

## 1. 常用命令

- `cd`
  - 切换目录
- `pwd`
  - 查看当前目录
- `ls`
  - 查看文件和目录
- `cp`
  - 复制文件或目录
- `mv`
  - 移动或重命名
- `rm`
  - 删除文件或目录
- `cat`
  - 查看文件内容
- `less`
  - 分页查看文件
- `grep`
  - 过滤匹配文本
- `ps`
  - 查看进程
- `top`
  - 动态查看系统状态
- `kill`
  - 给进程发信号
- `free`
  - 查看内存使用
- `tar`
  - 打包和解包

## 2. 查看进程和内存

### 查看进程状态

- `ps aux`
- `ps aux | grep xxx`
- `top`

### 查看内存使用

- `free -h`
- `free -m`
- `top`

## 3. `tar` 常见参数

- `-c`
  - 打包
- `-x`
  - 解包
- `-v`
  - 显示过程
- `-f`
  - 指定文件名
- `-z`
  - 配合 gzip
- `-j`
  - 配合 bzip2

常见例子：

```bash
tar -zcvf out.tar.gz dir/
tar -zxvf out.tar.gz
```

## 4. 文件权限修改

使用 `chmod`：

```bash
chmod 755 file
chmod -R 755 dir
```

权限位常见理解：

- `r`
  - 读
- `w`
  - 写
- `x`
  - 执行

三组身份：

- owner
- group
- others

## 5. 如何以 root 权限运行程序

常见方式：

- `sudo command`
- `su` 切到 root 后再执行

面试里通常答：

- 临时提权常用 `sudo`
- 切整个用户再执行可用 `su`

## 5.1 终端退出，终端里的进程会怎样

- 终端退出时，相关进程常会收到 `SIGHUP`
- 若程序没有特殊处理，通常也会退出

常见避免方式：

- `nohup command &`
- `setsid command`
- 使用 `screen` / `tmux`

## 6. 软链接和硬链接

### 软链接

- 本质上是一个保存目标路径的特殊文件
- 可跨文件系统
- 可链接文件或目录
- 原文件删掉后，软链接会失效

### 硬链接

- 本质上是给同一个 inode 增加一个目录项名字
- 一般不能跨文件系统
- 通常不能对目录创建硬链接
- 删除一个硬链接名，不影响其他同 inode 的名字

### 创建命令

```bash
ln source target      # 硬链接
ln -s source target   # 软链接
```

## 7. 静态库和动态库

### 常见文件名

- 静态库
  - `libxxx.a`
- 动态库
  - `libxxx.so`

### 区别

- 静态库
  - 链接时拷进可执行文件
  - 可执行文件更大
  - 运行时依赖更少
- 动态库
  - 运行时加载
  - 更节省磁盘和内存
  - 部署时要注意运行库环境

## 8. 静态库和动态库怎么制作

### 制作静态库

```bash
gcc -c hello.c -o hello.o
ar rcs libhello.a hello.o
```

### 制作动态库

```bash
gcc -fPIC -c hello.c -o hello.o
gcc -shared hello.o -o libhello.so
```

### 使用时常见方式

```bash
gcc main.c -L. -lhello -o app
```

## 9. 静态编译和动态编译

这个说法在很多面试资料里，其实常常混着“静态链接 / 动态链接”一起讲。

面试稳妥回答：

- `gcc test.c -o test.out`
  - 默认通常是动态链接
- `gcc -static test.c -o test.out`
  - 倾向静态链接

更准确地说：

- 编译阶段是把源码变成目标代码
- 链接阶段才会决定是静态链接还是动态链接

## 10. Python 和 C++ 的区别

- C++ 通常先编译成机器码再运行
- Python 通常先转成字节码，再由解释器执行
- C++ 更靠近底层，性能通常更高
- Python 开发效率和生态便利性通常更强

## 10.1 大端和小端

- 小端
  - 低有效字节放在低地址
- 大端
  - 高有效字节放在低地址

### 常见结论

- 很多主机平台常见是小端
- 网络字节序通常按大端约定

### 如何判断

- 可通过联合体、指针查看多字节整数的最低地址字节内容

## 10.2 网络通信时要不要做字节序转换

- 同一字节序平台之间通信，可能不需要额外转换
- 但跨平台网络通信时，通常应按网络字节序统一转换

常见函数：

- `htons`
- `htonl`
- `ntohs`
- `ntohl`

## 11. 面试速记

- 看进程：`ps aux` / `top`
- 看内存：`free -h`
- 解压：`tar -zxvf`
- 权限：`chmod`
- 提权：`sudo`
- 软硬链接：`ln -s` / `ln`
- 动态库：`.so`
- 静态库：`.a`
- 后台运行：`command &`
- 终端退出保活：`nohup command &`
