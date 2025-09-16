"""
智能家居系统 - 综合项目
集成传感器、执行器、通信协议的完整智能家居系统
"""

import machine
import time
import asyncio
import json
import network
import urequests
from machine import Pin, PWM, ADC, I2C, SPI

class SmartHomeController:
    """
    智能家居系统控制器
    """
    def __init__(self):
        """
        初始化智能家居系统
        """
        # 系统状态
        self.running = True
        self.mode = "auto"  # auto, manual, away
        self.temperature = 25.0
        self.humidity = 50.0
        self.light_level = 50.0
        self.motion_detected = False

        # 初始化硬件
        self.init_hardware()

        # 初始化通信
        self.init_communication()

        # 创建任务管理器
        self.task_manager = RTOSTaskManager()

        # 初始化设备管理器
        self.device_manager = DeviceManager()

        # 初始化自动化系统
        self.automation_system = AutomationSystem()

        # 系统配置
        self.config = {
            'temperature_range': (18.0, 26.0),
            'humidity_range': (30.0, 70.0),
            'light_threshold': 30.0,
            'motion_timeout': 300,  # 5分钟
            'server_url': 'http://your-server.com/api'
        }

    def init_hardware(self):
        """
        初始化硬件设备
        """
        # 输出设备
        self.led = Pin(25, Pin.OUT)
        self.relay1 = Pin(15, Pin.OUT)  # 空调控制
        self.relay2 = Pin(14, Pin.OUT)  # 加湿器控制
        self.relay3 = Pin(13, Pin.OUT)  # 灯光控制
        self.buzzer = Pin(12, Pin.OUT)

        # LED指示灯 (PWM用于亮度控制)
        self.status_led = PWM(Pin(16))
        self.status_led.freq(1000)
        self.status_led.duty_u16(0)

        # 输入设备
        self.temp_sensor = ADC(26)
        self.light_sensor = ADC(27)
        self.motion_sensor = Pin(18, Pin.IN)
        self.button = Pin(19, Pin.IN, Pin.PULL_UP)

        # 显示屏 (I2C OLED)
        self.i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=100000)
        self.display = I2CDisplay(self.i2c)

        # 串口通信
        self.uart = machine.UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))

        print("硬件初始化完成")

    def init_communication(self):
        """
        初始化通信系统
        """
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)

        # 传感器数据队列
        self.sensor_queue = asyncio.Queue(maxsize=10)
        self.command_queue = asyncio.Queue(maxsize=10)

    async def connect_wifi(self, ssid, password):
        """
        连接WiFi网络

        参数:
            ssid: WiFi名称
            password: WiFi密码
        """
        if not self.wlan.isconnected():
            print(f"连接WiFi: {ssid}")
            self.wlan.connect(ssid, password)

            # 等待连接
            for _ in range(10):
                if self.wlan.isconnected():
                    print(f"WiFi连接成功，IP: {self.wlan.ifconfig()[0]}")
                    return True
                await asyncio.sleep(1)

            print("WiFi连接失败")
            return False
        return True

    def read_temperature(self):
        """
        读取温度
        """
        # 模拟温度传感器 (NTC热敏电阻)
        adc_value = self.temp_sensor.read_u16()
        voltage = adc_value * 3.3 / 65535
        # 简化的温度计算
        self.temperature = 20 + (voltage - 1.65) * 10
        return self.temperature

    def read_light_level(self):
        """
        读取光照强度
        """
        adc_value = self.light_sensor.read_u16()
        self.light_level = (adc_value / 65535) * 100
        return self.light_level

    def read_motion_sensor(self):
        """
        读取运动传感器
        """
        motion = self.motion_sensor.value() == 1
        if motion:
            self.motion_detected = True
            self.last_motion_time = time.time()
        return self.motion_detected

    def check_motion_timeout(self):
        """
        检查运动超时
        """
        if self.motion_detected and hasattr(self, 'last_motion_time'):
            if time.time() - self.last_motion_time > self.config['motion_timeout']:
                self.motion_detected = False

    def set_relay(self, relay_num, state):
        """
        控制继电器

        参数:
            relay_num: 继电器编号 (1-3)
            state: 状态 (True/False)
        """
        if relay_num == 1:
            self.relay1.value(1 if state else 0)
        elif relay_num == 2:
            self.relay2.value(1 if state else 0)
        elif relay_num == 3:
            self.relay3.value(1 if state else 0)

    def set_led_brightness(self, brightness):
        """
        设置LED亮度

        参数:
            brightness: 亮度 (0-100)
        """
        duty = int(brightness * 65535 / 100)
        self.status_led.duty_u16(duty)

    def beep(self, duration=0.1, frequency=2000):
        """
        蜂鸣器发声

        参数:
            duration: 持续时间
            frequency: 频率
        """
        # 简单的蜂鸣器控制
        self.buzzer.on()
        time.sleep(duration)
        self.buzzer.off()

    def update_display(self):
        """
        更新显示屏
        """
        self.display.clear()
        self.display.text(f"Temp: {self.temperature:.1f}C", 0, 0)
        self.display.text(f"Light: {self.light_level:.0f}%", 0, 12)
        self.display.text(f"Mode: {self.mode}", 0, 24)
        self.display.text(f"Motion: {'Y' if self.motion_detected else 'N'}", 0, 36)
        self.display.show()

    async def sensor_monitor_task(self):
        """
        传感器监控任务
        """
        while self.running:
            # 读取传感器数据
            temp = self.read_temperature()
            light = self.read_light_level()
            motion = self.read_motion_sensor()

            # 检查运动超时
            self.check_motion_timeout()

            # 将数据放入队列
            sensor_data = {
                'temperature': temp,
                'humidity': self.humidity,
                'light': light,
                'motion': motion,
                'timestamp': time.time()
            }
            await self.sensor_queue.put(sensor_data)

            # 更新显示
            self.update_display()

            await asyncio.sleep(1)

    async def automation_task(self):
        """
        自动化控制任务
        """
        while self.running:
            if self.mode == "auto":
                # 温度控制
                if self.temperature < self.config['temperature_range'][0]:
                    self.set_relay(1, True)  # 开启加热
                elif self.temperature > self.config['temperature_range'][1]:
                    self.set_relay(1, False)  # 关闭加热

                # 湿度控制
                if self.humidity < self.config['humidity_range'][0]:
                    self.set_relay(2, True)  # 开启加湿
                elif self.humidity > self.config['humidity_range'][1]:
                    self.set_relay(2, False)  # 关闭加湿

                # 光照控制
                if self.light_level < self.config['light_threshold']:
                    if self.motion_detected:
                        self.set_relay(3, True)  # 开启灯光
                        self.set_led_brightness(100)
                    else:
                        self.set_relay(3, False)  # 关闭灯光
                        self.set_led_brightness(10)
                else:
                    self.set_relay(3, False)
                    self.set_led_brightness(0)

            elif self.mode == "away":
                # 离家模式：关闭所有设备
                self.set_relay(1, False)
                self.set_relay(2, False)
                self.set_relay(3, False)
                self.set_led_brightness(0)

            await asyncio.sleep(5)

    async def communication_task(self):
        """
        通信任务
        """
        while self.running:
            # 处理传感器数据
            if not self.sensor_queue.empty():
                sensor_data = await self.sensor_queue.get()
                await self.process_sensor_data(sensor_data)

            # 处理命令
            if not self.command_queue.empty():
                command = await self.command_queue.get()
                await self.process_command(command)

            # UART通信
            if self.uart.any():
                data = self.uart.read()
                if data:
                    command = json.loads(data.decode('utf-8'))
                    await self.command_queue.put(command)

            await asyncio.sleep(0.1)

    async def process_sensor_data(self, sensor_data):
        """
        处理传感器数据
        """
        # 检查报警条件
        if sensor_data['temperature'] > 30:
            self.beep(0.1, 2000)  # 高温报警

        # 发送到服务器
        if self.wlan.isconnected():
            try:
                response = urequests.post(
                    f"{self.config['server_url']}/sensor-data",
                    json=sensor_data,
                    timeout=5
                )
                if response.status_code == 200:
                    print("数据上传成功")
                response.close()
            except Exception as e:
                print(f"数据上传失败: {e}")

    async def process_command(self, command):
        """
        处理命令
        """
        cmd_type = command.get('type')

        if cmd_type == 'set_mode':
            self.mode = command.get('mode', 'auto')
            print(f"模式切换到: {self.mode}")

        elif cmd_type == 'control_device':
            device = command.get('device')
            state = command.get('state')
            if device == 'ac':
                self.set_relay(1, state)
            elif device == 'humidifier':
                self.set_relay(2, state)
            elif device == 'light':
                self.set_relay(3, state)

        elif cmd_type == 'get_status':
            status = {
                'mode': self.mode,
                'temperature': self.temperature,
                'humidity': self.humidity,
                'light': self.light_level,
                'motion': self.motion_detected,
                'devices': {
                    'ac': self.relay1.value() == 1,
                    'humidifier': self.relay2.value() == 1,
                    'light': self.relay3.value() == 1
                }
            }
            self.uart.write(json.dumps(status) + '\n')

    async def button_monitor_task(self):
        """
        按钮监控任务
        """
        button_pressed = False

        while self.running:
            if self.button.value() == 0:  # 按钮按下
                if not button_pressed:
                    button_pressed = True
                    # 切换模式
                    modes = ['auto', 'manual', 'away']
                    current_index = modes.index(self.mode)
                    self.mode = modes[(current_index + 1) % len(modes)]
                    print(f"模式切换到: {self.mode}")
                    self.beep(0.1, 1000)
            else:
                button_pressed = False

            await asyncio.sleep(0.1)

    async def run(self):
        """
        运行智能家居系统
        """
        print("智能家居系统启动")

        # 创建并启动所有任务
        tasks = [
            asyncio.create_task(self.sensor_monitor_task()),
            asyncio.create_task(self.automation_task()),
            asyncio.create_task(self.communication_task()),
            asyncio.create_task(self.button_monitor_task())
        ]

        # 运行系统
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("系统停止")
        finally:
            self.running = False
            # 清理资源
            self.set_relay(1, False)
            self.set_relay(2, False)
            self.set_relay(3, False)
            self.set_led_brightness(0)

class I2CDisplay:
    """
    简化的I2C OLED显示类
    """
    def __init__(self, i2c, addr=0x3C):
        """
        初始化显示屏

        参数:
            i2c: I2C对象
            addr: 显示屏地址
        """
        self.i2c = i2c
        self.addr = addr
        self.buffer = bytearray(128 * 8)  # 128x64显示屏

    def clear(self):
        """
        清空显示
        """
        self.buffer = bytearray(len(self.buffer))

    def text(self, text, x, y):
        """
        显示文本 (简化实现)
        """
        # 简化的文本显示，实际应用中需要完整字体
        if y < 64:
            for i, char in enumerate(text[:20]):
                if x + i < 128:
                    # 简单的字符显示逻辑
                    pass

    def show(self):
        """
        更新显示
        """
        # 发送显示数据到OLED
        # 这里简化实现，实际应用中需要完整的OLED驱动
        pass

class RTOSTaskManager:
    """
    简化的RTOS任务管理器
    """
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        """
        添加任务
        """
        self.tasks.append(task)

    def remove_task(self, task):
        """
        移除任务
        """
        if task in self.tasks:
            self.tasks.remove(task)

class DeviceManager:
    """
    设备管理器
    """
    def __init__(self):
        self.devices = {}

    def register_device(self, name, device):
        """
        注册设备
        """
        self.devices[name] = device

    def get_device(self, name):
        """
        获取设备
        """
        return self.devices.get(name)

class AutomationSystem:
    """
    自动化系统
    """
    def __init__(self):
        self.rules = []

    def add_rule(self, rule):
        """
        添加自动化规则
        """
        self.rules.append(rule)

    def evaluate_rules(self, sensor_data):
        """
        评估自动化规则
        """
        for rule in self.rules:
            rule.evaluate(sensor_data)

async def main():
    """
    主函数
    """
    print("=== 智能家居系统 ===")

    # 创建控制器
    controller = SmartHomeController()

    # 连接WiFi (需要根据实际情况修改)
    # await controller.connect_wifi("YourWiFi", "YourPassword")

    # 运行系统
    await controller.run()

if __name__ == "__main__":
    asyncio.run(main())