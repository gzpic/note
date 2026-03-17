---

---
- `xTaskCreate()` / `xTaskCreateStatic()` 创建两种方法
- `vTaskDelete()` 删任务
- `vTaskDelay()`, `vTaskDelayUntil()` 
- `vTaskStartScheduler()`
- `xTaskIncrementTick()`
- `vTaskSwitchContext()`
- `pxReadyTasksLists`、`xDelayedTaskList` 等链表结构

什么叫实时？

内部自己必须在一定的时间内完成自己规定的行为（时间既可以是某个时刻也可以是某个世家间隔）

对外部输入及时的做出正确的反馈

定时器硬件 (SysTick)
↓
Tick 中断触发 (1ms)
↓
xTaskIncrementTick()          ← 增加全局时间
↓
任务是否到期？               ← 检查延时队列
↓
是 → 移入就绪队列           ← 任务“被唤醒”
↓
vTaskSwitchContext()         ← 选择优先级最高的任务
↓
触发 PendSV 中断
↓
PendSV_Handler 汇编         ← 执行上下文切换
↓
新任务开始运行

外设产生中断（如 GPIO 按键）
↓
进入中断服务函数 ISR（如 EXTI_IRQHandler）
↓
ISR 中调用 xSemaphoreGiveFromISR() 或 xQueueSendFromISR()
↓
任务状态从“阻塞”变为“就绪” → 加入就绪队列
↓
判断是否要切换上下文（高优先级任务就绪？）
↓
若需要切换：设置 PendSV 中断挂起位（portYIELD_FROM_ISR）
↓
中断返回
↓
MCU 响应 PendSV 中断
↓
PendSV_Handler 汇编执行
↓
保存当前任务上下文，恢复新任务上下文
↓
新任务开始执行

```c
typedef struct tskTaskControlBlock       /* The old naming convention is used to prevent breaking kernel aware debuggers. */
{
ListItem_t xStateListItem;                  /< The list that the state list item of a task is reference from denotes the state of that task (Ready, Blocked, Suspended ). */
ListItem_t xEventListItem;                  /< Used to reference a task from an event list. */
UBaseType_t uxPriority;                     /< The priority of the task.  0 is the lowest priority. */
StackType_t * pxStack;                      /< Points to the start of the stack. */
volatile StackType_t * pxTopOfStack; /< Points to the location of the last item placed on the tasks stack.  THIS MUST BE THE FIRST MEMBER OF THE TCB STRUCT. */
char pcTaskName[ configMAX_TASK_NAME_LEN ]; /< Descriptive name given to the task when created.  Facilitates debugging only. */，
} tskTCB;
```

## 🧩 `volatile StackType_t * pxTopOfStack`

📌 **【任务栈顶指针 — 任务上下文的核心】**

- 表示当前任务的栈顶地址，也就是**任务上下文最后保存的位置**。
- 在任务切换时：
    - **切出时**：CPU 寄存器内容被保存到任务栈，并更新 `pxTopOfStack`。
    - **切入时**：根据 `pxTopOfStack` 恢复栈内容 → 恢复 CPU 状态 → 任务继续运行。
- 🔥 是 **TCB 中最重要的成员之一**，所以必须是结构体的**第一个成员**（有些架构中通过汇编操作偏移量为0的位置）。

---

## 📦 `ListItem_t xStateListItem`

📌 **【任务状态挂钩 — 就绪、阻塞、挂起】**

- 这个字段用于将任务挂入**各种状态链表**中，比如：
    - 就绪列表（ready list）
    - 延时列表（delay list）
    - 挂起列表（suspended list）
- 每次任务状态变更时，调度器通过这个字段把任务插入或移出相应链表。
- 是任务调度算法的关键结构。

---

## 🔗 `ListItem_t xEventListItem`

📌 **【事件等待挂钩 — 事件驱动时用】**

- 如果任务调用了 `xQueueReceive()` / `xSemaphoreTake()` 等等待操作，
就会把 `xEventListItem` 插入到某个**事件等待列表**中。
- 一旦事件触发（比如信号量可用），调度器会从该事件列表中把任务拉回到就绪列表。

---

## 🎚 `UBaseType_t uxPriority`

📌 **【任务当前优先级】**

- 表示当前任务的运行优先级，**数值越大优先级越高**。
- FreeRTOS 默认优先级从 `0`（最低）到 `(configMAX_PRIORITIES - 1)`。
- 任务调度器会始终优先选择优先级最高的“就绪任务”。

---

## 🧵 `StackType_t * pxStack`

📌 **【任务栈底指针】**

- 指向该任务栈的起始地址（栈底），一般用于：
    - 栈溢出检测
    - 栈使用统计
- 和 `pxTopOfStack` 配合使用，能够判断任务栈的使用量。
- 如果是静态任务，`pxStack` 会指向用户自己分配的栈区域。

---

## 🏷️ `char pcTaskName[ configMAX_TASK_NAME_LEN ]`

📌 **【任务名字 — 仅供调试】**

- 创建任务时设置的名字，存储为字符串。
- 不影响调度，仅用于调试和日志输出（例如在 Trace 工具、调试器里查看任务名）。

---

## 🧠 总结一下：

| 成员名 | 含义 | 作用 |
| --- | --- | --- |
| `pxTopOfStack` | 栈顶指针 | 保存/恢复任务上下文（任务切换核心） |
| `xStateListItem` | 状态挂钩 | 将任务挂入就绪/阻塞/挂起等状态链表 |
| `xEventListItem` | 事件挂钩 | 挂入等待事件的链表（等待信号量/消息队列） |
| `uxPriority` | 当前优先级 | 用于任务调度决策 |
| `pxStack` | 栈底地址 | 便于栈溢出检测和统计使用 |
| `pcTaskName` | 任务名 | 用于调试，无调度作用 |

- `pxTopOfStack` 是个指针，表示当前任务**栈顶（top）的位置**，**它是动态变化的**。
- `pxStack` 是个地址，表示任务**栈底（stack base）的位置**，**它在任务创建时就固定了**。


[[rtos链表]]

[[调度策略]]

[[tick]]