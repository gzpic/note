---

---

- `tasks.c`
    - `vTaskDelay`
    - `vTaskDelayUntil`
    - `prvAddCurrentTaskToDelayedList`
- `tasks.c`
    - `xTaskIncrementTick`（tick 更新）
    - 延时链表切换逻辑
- `list.c`
    - 链表操作函数，比如 `vListInsert`，理解延时任务是如何按唤醒时间排序插入的。
- `FreeRTOSConfig.h`
    - `configUSE_TICK_HOOK`、`configTICK_RATE_HZ` 等参数，决定时间管理精度和钩子函数。
- `port.c`（和具体移植平台相关）
    - 看 `SysTick_Handler` 或定时器中断怎么调用 `xTaskIncrementTick`。

## 它具体做了什么

1. **确定最高可运行优先级**
- 维护一个全局的最高就绪优先级 `uxTopReadyPriority`。
- 如果配置了 `configUSE_PORT_OPTIMISED_TASK_SELECTION=1`，用**位图**+端口内联宏（如 `portGET_HIGHEST_PRIORITY()`）O(1) 找到最高优先级；
- 否则在 `pxReadyTasksLists[0..configMAX_PRIORITIES-1]` 中自顶向下扫描第一个非空就绪链表。
2. **在该优先级就绪链表中选出“下一位”**
- 每个优先级有一个就绪链表（循环链表）。
- 通过 `listGET_OWNER_OF_NEXT_ENTRY()` 取得**下一个**结点的拥有者（TCB），并把链表**旋转一步**，实现**时间片轮转**（当 `configUSE_TIME_SLICING=1` 且有多个同优先级任务时）。
- 若同优先级只有 1 个任务，就会继续选它（无轮转）。
3. **更新当前任务指针**
- 将 `pxCurrentTCB` 设为被选中的 TCB（“下台/上台”的切换点）。
- 增加调度计数 `uxTaskNumber`（用于 trace/调试）。
4. **可选：埋点/钩子**
- 触发 `traceTASK_SWITCHED_IN()` / `traceTASK_SWITCHED_OUT()` 等宏，供 Tracealyzer/SystemView 等工具记录任务切换。
- 若启用了 `configUSE_NEWLIB_REENTRANT`，这里也会设置 newlib 的线程本地重入结构指针（`_impure_ptr`），保证 C 库可重入。

> 注意：它不做上下文保存/恢复；真正的寄存器压栈/出栈在端口层（PendSV_Handler 或等效汇编）完成，PendSV_Handler 在保存了旧任务上下文后调用 vTaskSwitchContext() 选出新任务，再恢复新任务上下文。

## 相关的数据结构与变量

- `pxReadyTasksLists[ configMAX_PRIORITIES ]`：按优先级划分的就绪循环链表，链表元素是任务的 `TCB_t`（通过其 `xStateListItem` 挂入）。
- `uxTopReadyPriority`：当前存在就绪任务的最高优先级。
- `pxCurrentTCB`：当前运行任务的 TCB 指针。

## 与其它机制的协作

- **时间片**：当同优先级多个任务就绪且 `configUSE_TIME_SLICING=1`，每次切换都会轮转到下一个任务，形成时间片轮转。
- **抢占**：当更高优先级任务变为就绪（通常在 `xTaskIncrementTick()` 或某个 `…FromISR` 唤醒后），`uxTopReadyPriority` 被提升，下一次切换必选更高优先级任务，实现抢占。
- **调用时机**：一般由 `PendSV` 路径触发（`SysTick` 或 ISR 末尾请求切换 → `PendSV` 里保存现场 → **调用 **`**vTaskSwitchContext()**`** 选人** → 恢复新任务）。