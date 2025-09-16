"""
RTOS任务管理示例
演示基于MicroPython的实时任务调度和管理
"""

import asyncio
import time
import machine

class RTOSTask:
    """
    RTOS任务基类
    """
    def __init__(self, name, priority=1, interval=1.0):
        """
        初始化任务

        参数:
            name: 任务名称
            priority: 任务优先级 (1-5, 5最高)
            interval: 任务执行间隔(秒)
        """
        self.name = name
        self.priority = priority
        self.interval = interval
        self.running = False
        self.task_handle = None

    async def setup(self):
        """
        任务初始化
        子类可重写此方法
        """
        pass

    async def execute(self):
        """
        任务执行逻辑
        子类必须重写此方法
        """
        raise NotImplementedError("子类必须实现execute方法")

    async def cleanup(self):
        """
        任务清理
        子类可重写此方法
        """
        pass

    async def run(self):
        """
        任务主循环
        """
        await self.setup()
        self.running = True

        try:
            while self.running:
                start_time = time.time()
                await self.execute()

                # 计算剩余时间
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)

                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            print(f"任务 {self.name} 被取消")
        except Exception as e:
            print(f"任务 {self.name} 出错: {e}")
        finally:
            await self.cleanup()

    def start(self):
        """
        启动任务
        """
        if not self.running:
            self.task_handle = asyncio.create_task(self.run())
            print(f"任务 {self.name} 已启动")

    def stop(self):
        """
        停止任务
        """
        if self.running:
            self.running = False
            if self.task_handle:
                self.task_handle.cancel()
            print(f"任务 {self.name} 已停止")

class LEDTask(RTOSTask):
    """
    LED控制任务
    """
    def __init__(self, led_pin=25, blink_pattern=(0.5, 0.5), **kwargs):
        super().__init__(**kwargs)
        self.led = machine.Pin(led_pin, machine.Pin.OUT)
        self.blink_pattern = blink_pattern  # (on_time, off_time)

    async def execute(self):
        """
        执行LED闪烁
        """
        self.led.on()
        await asyncio.sleep(self.blink_pattern[0])
        self.led.off()
        await asyncio.sleep(self.blink_pattern[1])

class SensorTask(RTOSTask):
    """
    传感器读取任务
    """
    def __init__(self, sensor_pin=26, **kwargs):
        super().__init__(**kwargs)
        self.adc = machine.ADC(sensor_pin)

    async def execute(self):
        """
        读取传感器数据
        """
        # 模拟传感器读取
        value = self.adc.read_u16()
        voltage = value * 3.3 / 65535
        print(f"传感器值: {value}, 电压: {voltage:.2f}V")

class ButtonTask(RTOSTask):
    """
    按钮监控任务
    """
    def __init__(self, button_pin=16, **kwargs):
        super().__init__(**kwargs)
        self.button = machine.Pin(button_pin, machine.Pin.IN, machine.Pin.PULL_UP)
        self.last_state = 1
        self.press_count = 0

    async def execute(self):
        """
        监控按钮状态
        """
        current_state = self.button.value()

        if current_state != self.last_state:
            if current_state == 0:  # 按钮按下
                self.press_count += 1
                print(f"按钮按下 (第{self.press_count}次)")
            self.last_state = current_state

class RTOSManager:
    """
    RTOS管理器
    """
    def __init__(self):
        self.tasks = []
        self.running = False

    def add_task(self, task):
        """
        添加任务
        """
        self.tasks.append(task)
        # 按优先级排序
        self.tasks.sort(key=lambda x: x.priority, reverse=True)

    def remove_task(self, task_name):
        """
        移除任务
        """
        self.tasks = [t for t in self.tasks if t.name != task_name]

    def start_all(self):
        """
        启动所有任务
        """
        self.running = True
        for task in self.tasks:
            task.start()

    def stop_all(self):
        """
        停止所有任务
        """
        self.running = False
        for task in self.tasks:
            task.stop()

    def get_task_status(self):
        """
        获取所有任务状态
        """
        status = []
        for task in self.tasks:
            status.append({
                'name': task.name,
                'priority': task.priority,
                'running': task.running,
                'interval': task.interval
            })
        return status

    def print_status(self):
        """
        打印任务状态
        """
        print("\n=== 任务状态 ===")
        for task in self.tasks:
            status = "运行中" if task.running else "已停止"
            print(f"{task.name}: 优先级{task.priority}, {status}")
        print("=" * 20)

class CommunicationQueue:
    """
    任务间通信队列
    """
    def __init__(self, maxsize=10):
        self.queue = asyncio.Queue(maxsize)

    async def put(self, message):
        """
        发送消息
        """
        await self.queue.put(message)

    async def get(self):
        """
        接收消息
        """
        return await self.queue.get()

    def qsize(self):
        """
        队列大小
        """
        return self.queue.qsize()

class DataProcessingTask(RTOSTask):
    """
    数据处理任务
    """
    def __init__(self, comm_queue, **kwargs):
        super().__init__(**kwargs)
        self.comm_queue = comm_queue

    async def execute(self):
        """
        处理队列中的数据
        """
        try:
            # 非阻塞方式获取消息
            message = await asyncio.wait_for(self.comm_queue.get(), timeout=0.1)
            print(f"处理消息: {message}")

            # 模拟数据处理
            if isinstance(message, dict) and 'temperature' in message:
                temp = message['temperature']
                if temp > 25:
                    print("警告: 温度过高!")
        except asyncio.TimeoutError:
            pass

async def main():
    """
    主函数 - 演示RTOS任务管理
    """
    print("=== RTOS任务管理示例 ===")

    # 创建RTOS管理器
    manager = RTOSManager()

    # 创建通信队列
    comm_queue = CommunicationQueue()

    # 创建并添加任务
    led_task = LEDTask(
        name="LED闪烁",
        priority=3,
        interval=1.0,
        blink_pattern=(0.3, 0.7)
    )

    sensor_task = SensorTask(
        name="温度传感器",
        priority=2,
        interval=2.0
    )

    button_task = ButtonTask(
        name="按钮监控",
        priority=4,
        interval=0.1
    )

    processing_task = DataProcessingTask(
        name="数据处理",
        priority=1,
        interval=0.5,
        comm_queue=comm_queue
    )

    # 添加任务到管理器
    manager.add_task(led_task)
    manager.add_task(sensor_task)
    manager.add_task(button_task)
    manager.add_task(processing_task)

    # 显示初始状态
    manager.print_status()

    # 启动所有任务
    manager.start_all()

    # 模拟数据生产
    async def produce_data():
        """数据生产者"""
        count = 0
        while manager.running:
            count += 1
            message = {
                'temperature': 20 + (count % 10),
                'timestamp': time.time()
            }
            await comm_queue.put(message)
            await asyncio.sleep(3)

    # 启动数据生产
    producer_task = asyncio.create_task(produce_data())

    # 主循环
    try:
        while True:
            # 每隔5秒显示状态
            await asyncio.sleep(5)
            manager.print_status()

    except KeyboardInterrupt:
        print("\n接收到停止信号")
    finally:
        # 停止所有任务
        manager.stop_all()
        producer_task.cancel()

        # 等待任务完成
        await asyncio.sleep(1)
        print("RTOS系统已停止")

if __name__ == "__main__":
    asyncio.run(main())