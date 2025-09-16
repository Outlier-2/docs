"""
环境监测站项目
完整的环境监测系统，包含多种传感器、数据记录和报警功能
"""

import machine
import time
import asyncio
import json
import struct
import math
from machine import Pin, ADC, I2C, SPI, PWM

class EnvironmentMonitor:
    """
    环境监测站主控制器
    """
    def __init__(self):
        """
        初始化环境监测站
        """
        # 系统状态
        self.running = True
        self.recording = False
        self.alerts_enabled = True

        # 传感器数据
        self.sensor_data = {
            'temperature': 0.0,
            'humidity': 0.0,
            'pressure': 0.0,
            'light': 0.0,
            'air_quality': 0.0,
            'sound_level': 0.0,
            'motion': False,
            'timestamp': 0
        }

        # 初始化硬件
        self.init_hardware()

        # 初始化传感器
        self.init_sensors()

        # 初始化存储
        self.init_storage()

        # 初始化报警系统
        self.init_alert_system()

        # 报警阈值
        self.alert_thresholds = {
            'temperature': (15.0, 35.0),
            'humidity': (20.0, 80.0),
            'air_quality': 500,  # AQI阈值
            'sound_level': 800   # 噪声阈值
        }

        print("环境监测站初始化完成")

    def init_hardware(self):
        """
        初始化硬件
        """
        # LED指示灯
        self.status_led = Pin(25, Pin.OUT)
        self.alert_led = Pin(15, Pin.OUT)
        self.recording_led = Pin(14, Pin.OUT)

        # 蜂鸣器
        self.buzzer = Pin(13, Pin.OUT)

        # 按钮
        self.record_button = Pin(16, Pin.IN, Pin.Pin.PULL_UP)
        self.alert_button = Pin(17, Pin.IN, Pin.Pin.PULL_UP)
        self.mode_button = Pin(18, Pin.IN, Pin.Pin.PULL_UP)

        # 显示屏
        self.i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=100000)
        self.display = DisplayDriver(self.i2c)

        # SD卡 (SPI)
        self.spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
                       sck=Pin(2), mosi=Pin(3), miso=Pin(4))
        self.cs = Pin(5, Pin.OUT)
        self.cs.value(1)

        print("硬件初始化完成")

    def init_sensors(self):
        """
        初始化传感器
        """
        # 温湿度传感器 (模拟DHT22)
        self.dht_sensor = DHTSensor(pin=6)

        # 气压传感器 (模拟BMP280)
        self.bmp_sensor = BMPSensor(i2c=self.i2c, addr=0x76)

        # 光敏传感器
        self.light_sensor = LightSensor(pin=26)

        # 空气质量传感器
        self.air_sensor = AirQualitySensor(pin=27)

        # 声音传感器
        self.sound_sensor = SoundSensor(pin=28)

        # 运动传感器
        self.motion_sensor = MotionSensor(pin=19)

        print("传感器初始化完成")

    def init_storage(self):
        """
        初始化存储系统
        """
        self.data_logger = DataLogger(max_records=1000)
        self.file_storage = FileStorage(spi=self.spi, cs_pin=self.cs)

    def init_alert_system(self):
        """
        初始化报警系统
        """
        self.alert_manager = AlertManager(
            led_pin=15,
            buzzer_pin=13
        )

    def read_all_sensors(self):
        """
        读取所有传感器数据
        """
        try:
            # 读取温湿度
            temp, humidity = self.dht_sensor.read()
            self.sensor_data['temperature'] = temp
            self.sensor_data['humidity'] = humidity

            # 读取气压
            pressure = self.bmp_sensor.read_pressure()
            self.sensor_data['pressure'] = pressure

            # 读取光照
            light = self.light_sensor.read()
            self.sensor_data['light'] = light

            # 读取空气质量
            air_quality = self.air_sensor.read()
            self.sensor_data['air_quality'] = air_quality

            # 读取声音
            sound = self.sound_sensor.read()
            self.sensor_data['sound_level'] = sound

            # 读取运动
            motion = self.motion_sensor.read()
            self.sensor_data['motion'] = motion

            # 时间戳
            self.sensor_data['timestamp'] = time.time()

            return True

        except Exception as e:
            print(f"传感器读取错误: {e}")
            return False

    def check_alerts(self):
        """
        检查报警条件
        """
        if not self.alerts_enabled:
            return

        alerts = []

        # 温度报警
        if (self.sensor_data['temperature'] < self.alert_thresholds['temperature'][0] or
            self.sensor_data['temperature'] > self.alert_thresholds['temperature'][1]):
            alerts.append({
                'type': 'temperature',
                'message': f"温度异常: {self.sensor_data['temperature']:.1f}°C",
                'severity': 'high' if abs(self.sensor_data['temperature'] - 25) > 10 else 'medium'
            })

        # 湿度报警
        if (self.sensor_data['humidity'] < self.alert_thresholds['humidity'][0] or
            self.sensor_data['humidity'] > self.alert_thresholds['humidity'][1]):
            alerts.append({
                'type': 'humidity',
                'message': f"湿度异常: {self.sensor_data['humidity']:.1f}%",
                'severity': 'medium'
            })

        # 空气质量报警
        if self.sensor_data['air_quality'] > self.alert_thresholds['air_quality']:
            alerts.append({
                'type': 'air_quality',
                'message': f"空气质量差: {self.sensor_data['air_quality']}",
                'severity': 'high'
            })

        # 声音报警
        if self.sensor_data['sound_level'] > self.alert_thresholds['sound_level']:
            alerts.append({
                'type': 'sound_level',
                'message': f"噪声超标: {self.sensor_data['sound_level']}",
                'severity': 'low'
            })

        # 触发报警
        for alert in alerts:
            self.alert_manager.trigger_alert(alert)

    def update_display(self):
        """
        更新显示屏
        """
        self.display.clear()

        # 显示主要数据
        self.display.text(f"Temp: {self.sensor_data['temperature']:.1f}C", 0, 0)
        self.display.text(f"Hum: {self.sensor_data['humidity']:.1f}%", 0, 12)
        self.display.text(f"Pres: {self.sensor_data['pressure']:.0f}hPa", 0, 24)
        self.display.text(f"Air: {self.sensor_data['air_quality']:.0f}", 0, 36)

        # 显示状态
        status_text = "REC" if self.recording else "STOP"
        alert_text = "ALERT" if self.alert_manager.has_active_alerts() else "OK"

        self.display.text(f"{status_text} {alert_text}", 0, 48)

        self.display.show()

    async def sensor_monitor_task(self):
        """
        传感器监控任务
        """
        while self.running:
            if self.read_all_sensors():
                # 记录数据
                if self.recording:
                    self.data_logger.add_record(self.sensor_data.copy())

                # 检查报警
                self.check_alerts()

                # 更新显示
                self.update_display()

            await asyncio.sleep(2)

    async def button_monitor_task(self):
        """
        按钮监控任务
        """
        record_pressed = False
        alert_pressed = False
        mode_pressed = False

        while self.running:
            # 记录按钮
            if self.record_button.value() == 0:
                if not record_pressed:
                    record_pressed = True
                    self.recording = not self.recording
                    self.recording_led.value(1 if self.recording else 0)
                    print(f"记录状态: {'开启' if self.recording else '关闭'}")
            else:
                record_pressed = False

            # 报警按钮
            if self.alert_button.value() == 0:
                if not alert_pressed:
                    alert_pressed = True
                    self.alerts_enabled = not self.alerts_enabled
                    print(f"报警状态: {'开启' if self.alerts_enabled else '关闭'}")
            else:
                alert_pressed = False

            # 模式按钮
            if self.mode_button.value() == 0:
                if not mode_pressed:
                    mode_pressed = True
                    # 切换显示模式
                    self.display.cycle_mode()
            else:
                mode_pressed = False

            await asyncio.sleep(0.1)

    async def data_storage_task(self):
        """
        数据存储任务
        """
        while self.running:
            # 定期保存数据到文件
            if self.recording:
                records = self.data_logger.get_records_since_last_save()
                if records:
                    for record in records:
                        self.file_storage.append_record(record)
                    print(f"保存了 {len(records)} 条记录")

            await asyncio.sleep(60)  # 每分钟保存一次

    async def alert_monitor_task(self):
        """
        报警监控任务
        """
        while self.running:
            # 更新报警状态
            self.alert_manager.update()

            # LED状态更新
            if self.alert_manager.has_active_alerts():
                self.alert_led.on()
            else:
                self.alert_led.off()

            await asyncio.sleep(1)

    async def data_analysis_task(self):
        """
        数据分析任务
        """
        while self.running:
            # 每小时分析一次数据
            await asyncio.sleep(3600)

            # 分析趋势
            analysis = self.data_logger.analyze_trends()

            # 显示分析结果
            print("=== 数据分析结果 ===")
            for sensor, trend in analysis.items():
                print(f"{sensor}: {trend}")

            # 检查异常模式
            anomalies = self.data_logger.detect_anomalies()
            if anomalies:
                print("检测到异常模式:")
                for anomaly in anomalies:
                    print(f"  {anomaly}")

    async def run(self):
        """
        运行环境监测站
        """
        print("环境监测站启动")

        # 创建所有任务
        tasks = [
            asyncio.create_task(self.sensor_monitor_task()),
            asyncio.create_task(self.button_monitor_task()),
            asyncio.create_task(self.data_storage_task()),
            asyncio.create_task(self.alert_monitor_task()),
            asyncio.create_task(self.data_analysis_task())
        ]

        # 运行系统
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            print("环境监测站停止")
        finally:
            self.running = False

            # 保存剩余数据
            if self.recording:
                records = self.data_logger.get_all_records()
                for record in records:
                    self.file_storage.append_record(record)

            # 关闭硬件
            self.status_led.off()
            self.alert_led.off()
            self.recording_led.off()
            self.buzzer.off()

# 传感器类
class DHTSensor:
    """温湿度传感器"""
    def __init__(self, pin):
        self.pin = pin
        self.last_reading = (0.0, 0.0)

    def read(self):
        """读取温湿度"""
        # 模拟DHT22读取
        import random
        temp = 20 + random.uniform(-5, 15)
        humidity = 50 + random.uniform(-20, 30)
        self.last_reading = (temp, humidity)
        return temp, humidity

class BMPSensor:
    """气压传感器"""
    def __init__(self, i2c, addr=0x76):
        self.i2c = i2c
        self.addr = addr

    def read_pressure(self):
        """读取气压"""
        # 模拟BMP280读取
        import random
        return 1013 + random.uniform(-20, 20)

class LightSensor:
    """光敏传感器"""
    def __init__(self, pin):
        self.adc = ADC(pin)

    def read(self):
        """读取光照强度"""
        value = self.adc.read_u16()
        return (value / 65535) * 100

class AirQualitySensor:
    """空气质量传感器"""
    def __init__(self, pin):
        self.adc = ADC(pin)

    def read(self):
        """读取空气质量"""
        value = self.adc.read_u16()
        return int(value / 65535 * 1000)

class SoundSensor:
    """声音传感器"""
    def __init__(self, pin):
        self.adc = ADC(pin)

    def read(self):
        """读取声音级别"""
        samples = []
        for _ in range(10):
            samples.append(self.adc.read_u16())
            time.sleep_ms(1)
        return sum(samples) // len(samples)

class MotionSensor:
    """运动传感器"""
    def __init__(self, pin):
        self.pin = Pin(pin, Pin.IN)

    def read(self):
        """读取运动状态"""
        return self.pin.value() == 1

# 显示驱动类
class DisplayDriver:
    """显示屏驱动"""
    def __init__(self, i2c, addr=0x3C):
        self.i2c = i2c
        self.addr = addr
        self.mode = 0
        self.modes = ['main', 'detail', 'graph']

    def clear(self):
        """清空显示"""
        pass

    def text(self, text, x, y):
        """显示文本"""
        pass

    def show(self):
        """更新显示"""
        pass

    def cycle_mode(self):
        """切换显示模式"""
        self.mode = (self.mode + 1) % len(self.modes)

# 数据记录类
class DataLogger:
    """数据记录器"""
    def __init__(self, max_records=1000):
        self.max_records = max_records
        self.records = []
        self.last_save_index = 0

    def add_record(self, record):
        """添加记录"""
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def get_records_since_last_save(self):
        """获取上次保存后的记录"""
        records = self.records[self.last_save_index:]
        self.last_save_index = len(self.records)
        return records

    def get_all_records(self):
        """获取所有记录"""
        return self.records

    def analyze_trends(self):
        """分析趋势"""
        if len(self.records) < 10:
            return {}

        # 简单的趋势分析
        trends = {}
        for key in ['temperature', 'humidity', 'pressure']:
            values = [r[key] for r in self.records[-10:]]
            avg = sum(values) / len(values)
            trends[key] = f"平均: {avg:.1f}"

        return trends

    def detect_anomalies(self):
        """检测异常"""
        if len(self.records) < 20:
            return []

        anomalies = []
        for key in ['temperature', 'humidity']:
            values = [r[key] for r in self.records[-20:]]
            avg = sum(values) / len(values)
            std = math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))

            # 检查超出2个标准差的值
            for i, value in enumerate(values):
                if abs(value - avg) > 2 * std:
                    anomalies.append(f"{key}异常: {value:.1f} (平均: {avg:.1f})")

        return anomalies

# 文件存储类
class FileStorage:
    """文件存储"""
    def __init__(self, spi, cs_pin):
        self.spi = spi
        self.cs = Pin(cs_pin, Pin.OUT)
        self.filename = "env_data.txt"

    def append_record(self, record):
        """追加记录"""
        # 简化的文件写入
        line = json.dumps(record) + "\n"
        # 实际应用中需要写入文件系统
        print(f"保存记录: {line.strip()}")

# 报警管理类
class AlertManager:
    """报警管理器"""
    def __init__(self, led_pin, buzzer_pin):
        self.led = Pin(led_pin, Pin.OUT)
        self.buzzer = Pin(buzzer_pin, Pin.OUT)
        self.active_alerts = []

    def trigger_alert(self, alert):
        """触发报警"""
        print(f"报警: {alert['message']}")
        self.active_alerts.append(alert)

        # LED闪烁
        for _ in range(3):
            self.led.on()
            time.sleep(0.1)
            self.led.off()
            time.sleep(0.1)

        # 蜂鸣器
        if alert['severity'] == 'high':
            self.buzzer.on()
            time.sleep(0.5)
            self.buzzer.off()

    def has_active_alerts(self):
        """检查是否有活动报警"""
        return len(self.active_alerts) > 0

    def update(self):
        """更新报警状态"""
        # 清理过期的报警
        current_time = time.time()
        self.active_alerts = [
            alert for alert in self.active_alerts
            if current_time - alert.get('timestamp', 0) < 300  # 5分钟
        ]

async def main():
    """
    主函数
    """
    print("=== 环境监测站 ===")

    # 创建监测站
    monitor = EnvironmentMonitor()

    # 运行系统
    await monitor.run()

if __name__ == "__main__":
    asyncio.run(main())