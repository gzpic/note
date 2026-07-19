---

---
## ✅ 唯一按优先级划分的队列：

### 🔸 `pxReadyTasksLists[]`

- 类型：数组（`List_t pxReadyTasksLists[ configMAX_PRIORITIES ];`）
- 含义：每个优先级都有一个“就绪任务链表”
- ✅ 每个优先级一个，调度器从高优先级向下遍历这个数组来选任务

---

## ❌ 其余都是**全局唯一的队列**：

| 队列名称 | 独立多个？ | 用途 | 优先级有关？ |
| --- | --- | --- | --- |
| `xDelayedTaskList1` / `xDelayedTaskList2` | ❌ 否，全局各一个 | 延时等待 | ❌ 无关，按唤醒时间排序 |
| `xPendingReadyList` | ❌ 否，全局一个 | 中断中唤醒的任务缓存区 | ✅ 用于快速转入就绪队列，但自身不分优先级 |
| `xSuspendedTaskList` | ❌ 否，全局一个 | 被挂起的任务 | ❌ 无关，人工控制 |
| `xTasksWaitingTermination` | ❌ 否，全局一个 | 已删除任务等待清理 | ❌ 无关，仅资源清理 |
| `xBlockedTaskList`（概念） | ❌ 否，散布在各资源对象中 | 被阻塞等待队列/信号量 | ❌ 只与资源绑定，不按优先级分类 |

---

## 🧠 为什么只有就绪队列按优先级划分？

因为调度器需要：

- 快速判断**当前是否有更高优先级任务可以运行**
- 效率地进行任务选择：从 `pxReadyTasksLists[]` 中按优先级向下找第一个非空链表

而其他队列：

- **不参与调度器直接选任务的过程**
- 只是状态性的挂载点（延时/挂起/阻塞/终止）
- 不需要频繁查找“优先级最高任务”

---

## ✅ 总结一句话：

> FreeRTOS 中只有就绪队列 pxReadyTasksLists[] 是按任务优先级划分的，其它如延时、挂起、终止等任务队列都是全局共享、优先级无关的链表结构。

## **① 任务状态相关链表（调度器核心）**

这些直接决定任务状态、调度顺序：

| 链表 | 作用 | 节点字段 |
| --- | --- | --- |
| `**pxReadyTasksLists[configMAX_PRIORITIES]**` | 按优先级存放就绪任务 | `xStateListItem` |
| `**pxDelayedTaskList**`** / **`**pxOverflowDelayedTaskList**` | 延时/阻塞任务，按唤醒 tick 升序 | `xStateListItem` |
| `**xSuspendedTaskList**` | 主动挂起的任务 | `xStateListItem` |
| `**xTasksWaitingTermination**` | 已删除但等待 idle 清理的任务 | `xStateListItem` |
| `**xPendingReadyList**` | 调度器挂起期间被中断唤醒的任务（临时队列） | `xStateListItem` |

---

## **② 非任务状态相关链表（内核对象 / 其他子系统）**

这些链表不直接反映任务全局状态，但用于同步、定时等功能：

| 链表 | 作用 | 节点字段 |
| --- | --- | --- |
| **队列的等待链表** `xTasksWaitingToSend` / `xTasksWaitingToReceive` | 队列满/空时等待的任务 | `xEventListItem` |
| **事件组等待链表** `xTasksWaitingForBits` | 等待事件组特定位条件的任务 | `xEventListItem` |
| **流缓冲区等待链表**（可选） | 等待缓冲数据或空间的任务 | `xEventListItem` |
| `**xActiveTimerList**`（软件定时器启用时） | 按到期时间排序的活动定时器 | `xTimerListItem` |
| **内存空闲块链表**（heap_4/5） | 管理堆上空闲内存块 | `BlockLink_t` |