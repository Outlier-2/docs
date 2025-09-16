# 嵌入式系统设计 - 第三讲：实时操作系统概念

## 实时操作系统概述

### 什么是RTOS？
实时操作系统（Real-Time Operating System）是专门为嵌入式系统设计的操作系统，能够保证任务在确定的时间内完成。

### RTOS的特点
- **实时性**: 任务必须在严格的时间限制内完成
- **可预测性**: 系统行为可以预测
- **可靠性**: 系统必须稳定运行
- **资源管理**: 有效管理系统资源

### RTOS vs 通用操作系统
| 特性 | RTOS | 通用操作系统 |
|------|------|-------------|
| 响应时间 | 微秒级 | 毫秒级 |
| 调度策略 | 优先级调度 | 时间片轮转 |
| 内存使用 | 小 | 大 |
| 实时性 | 硬实时/软实时 | 非实时 |

## MicroPython中的并发编程

### 异步编程基础
MicroPython支持异步编程，可以实现类似RTOS的功能。

```python
import asyncio
import time

async def blink_led(interval):
    """LED闪烁任务"""
    led = machine.Pin(25, machine.Pin.OUT)
    while True:
        led.on()
        await asyncio.sleep(interval)
        led.off()
        await asyncio.sleep(interval)

async def read_sensor():
    """传感器读取任务"""
    while True:
        # 模拟传感器读取
        print("读取传感器数据")
        await asyncio.sleep(1)

async def main():
    """主任务"""
    # 创建任务
    task1 = asyncio.create_task(blink_led(0.5))
    task2 = asyncio.create_task(read_sensor())

    # 等待任务完成
    await asyncio.gather(task1, task2)

# 运行异步程序
asyncio.run(main())
```

## 任务调度

### 协程和任务
```python
import asyncio

async def task1():
    """任务1"""
    for i in range(5):
        print(f"任务1: {i}")
        await asyncio.sleep(0.1)

async def task2():
    """任务2"""
    for i in range(5):
        print(f"任务2: {i}")
        await asyncio.sleep(0.2)

async def main():
    """主程序"""
    # 并发执行任务
    await asyncio.gather(task1(), task2())
```

### 优先级调度
```python
import asyncio
import time

class PriorityTask:
    """优先级任务"""
    def __init__(self, name, priority, duration):
        self.name = name
        self.priority = priority
        self.duration = duration

    async def run(self):
        """运行任务"""
        start_time = time.time()
        while time.time() - start_time < self.duration:
            print(f"任务 {self.name} (优先级: {self.priority})")
            await asyncio.sleep(0.1)

class SimpleScheduler:
    """简单调度器"""
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        """添加任务"""
        self.tasks.append(task)
        # 按优先级排序
        self.tasks.sort(key=lambda x: x.priority, reverse=True)

    async def run(self):
        """运行调度器"""
        while True:
            for task in self.tasks:
                await task.run()
            await asyncio.sleep(0.01)

# 使用示例
async def main():
    scheduler = SimpleScheduler()

    # 添加不同优先级的任务
    scheduler.add_task(PriorityTask("高优先级", 3, 2))
    scheduler.add_task(PriorityTask("中优先级", 2, 2))
    scheduler.add_task(PriorityTask("低优先级", 1, 2))

    await scheduler.run()

asyncio.run(main())
```

## 同步和互斥

### 信号量
```python
import asyncio

class Semaphore:
    """简单信号量实现"""
    def __init__(self, permits=1):
        self.permits = permits
        self.queue = asyncio.Queue()

    async def acquire(self):
        """获取信号量"""
        if self.permits > 0:
            self.permits -= 1
            return True
        else:
            # 等待信号量
            event = asyncio.Event()
            await self.queue.put(event)
            await event.wait()
            return True

    def release(self):
        """释放信号量"""
        if not self.queue.empty():
            # 唤醒等待的任务
            event = self.queue.get_nowait()
            event.set()
        else:
            self.permits += 1

# 使用示例
async def worker(name, semaphore):
    """工作者任务"""
    await semaphore.acquire()
    print(f"工作者 {name} 获得信号量")
    await asyncio.sleep(1)
    print(f"工作者 {name} 释放信号量")
    semaphore.release()

async def main():
    semaphore = Semaphore(2)  # 最多2个并发

    # 创建多个工作者
    workers = [
        asyncio.create_task(worker(f"Worker{i}", semaphore))
        for i in range(5)
    ]

    await asyncio.gather(*workers)

asyncio.run(main())
```

### 队列通信
```python
import asyncio

class MessageQueue:
    """消息队列"""
    def __init__(self, maxsize=10):
        self.queue = asyncio.Queue(maxsize)

    async def put(self, item):
        """放入消息"""
        await self.queue.put(item)

    async def get(self):
        """获取消息"""
        return await self.queue.get()

    def qsize(self):
        """队列大小"""
        return self.queue.qsize()

# 生产者-消费者模式
async def producer(queue, id):
    """生产者"""
    for i in range(5):
        message = f"生产者{id}-消息{i}"
        await queue.put(message)
        print(f"生产: {message}")
        await asyncio.sleep(0.1)

async def consumer(queue):
    """消费者"""
    while True:
        message = await queue.get()
        print(f"消费: {message}")
        await asyncio.sleep(0.2)

async def main():
    queue = MessageQueue()

    # 创建生产者和消费者
    producers = [
        asyncio.create_task(producer(queue, i))
        for i in range(2)
    ]
    consumer_task = asyncio.create_task(consumer(queue))

    await asyncio.gather(*producers)

    # 等待队列清空
    while not queue.qsize() == 0:
        await asyncio.sleep(0.1)

    # 取消消费者任务
    consumer_task.cancel()

asyncio.run(main())
```

## 实际应用示例

### 多任务传感器系统
```python
import asyncio
import machine
import time

class SensorSystem:
    """传感器系统"""
    def __init__(self):
        self.led = machine.Pin(25, machine.Pin.OUT)
        self.button = machine.Pin(16, machine.Pin.IN)
        self.running = True

        # 通信队列
        self.sensor_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()

    async def read_temperature(self):
        """读取温度传感器"""
        while self.running:
            # 模拟温度读取
            temp = 20 + (time.time() % 10)
            await self.sensor_queue.put(("temperature", temp))
            await asyncio.sleep(1)

    async def read_humidity(self):
        """读取湿度传感器"""
        while self.running:
            # 模拟湿度读取
            humidity = 50 + (time.time() % 20)
            await self.sensor_queue.put(("humidity", humidity))
            await asyncio.sleep(2)

    async def process_data(self):
        """处理传感器数据"""
        while self.running:
            try:
                sensor_type, value = await asyncio.wait_for(
                    self.sensor_queue.get(), timeout=1.0
                )
                print(f"传感器 {sensor_type}: {value:.1f}")

                # 检查报警条件
                if sensor_type == "temperature" and value > 28:
                    await self.alert_queue.put(("high_temp", value))

            except asyncio.TimeoutError:
                continue

    async def handle_alerts(self):
        """处理报警"""
        while self.running:
            try:
                alert_type, value = await asyncio.wait_for(
                    self.alert_queue.get(), timeout=1.0
                )
                print(f"报警: {alert_type} = {value}")

                # LED闪烁报警
                for _ in range(3):
                    self.led.on()
                    await asyncio.sleep(0.1)
                    self.led.off()
                    await asyncio.sleep(0.1)

            except asyncio.TimeoutError:
                continue

    async def monitor_button(self):
        """监控按钮"""
        while self.running:
            if self.button.value() == 0:
                print("按钮按下，停止系统")
                self.running = False
                break
            await asyncio.sleep(0.1)

    async def run(self):
        """运行系统"""
        print("传感器系统启动")

        # 创建所有任务
        tasks = [
            asyncio.create_task(self.read_temperature()),
            asyncio.create_task(self.read_humidity()),
            asyncio.create_task(self.process_data()),
            asyncio.create_task(self.handle_alerts()),
            asyncio.create_task(self.monitor_button())
        ]

        # 等待所有任务完成
        await asyncio.gather(*tasks)

        print("传感器系统停止")

# 使用示例
async def main():
    system = SensorSystem()
    await system.run()

asyncio.run(main())
```

## 实践练习

### 练习1：多任务LED控制
- 创建多个LED控制任务
- 实现不同的闪烁模式
- 使用信号量控制访问

### 练习2：数据采集系统
- 实现多传感器数据采集
- 使用队列进行数据传输
- 实现数据处理和报警

### 练习3：优先级调度器
- 实现简单的优先级调度器
- 测试不同优先级任务的执行
- 观察调度行为

## 课后作业
1. 实现一个完整的多任务嵌入式系统
2. 添加任务间通信机制
3. 实现优先级调度算法

## 下一讲预告
传感器接口和数据采集