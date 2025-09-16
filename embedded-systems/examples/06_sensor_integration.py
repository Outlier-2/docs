"""
传感器集成示例
演示多种传感器数据采集和处理
"""

import machine
import time
import asyncio
import json

class SensorBase:
    """
    传感器基类
    """
    def __init__(self, name, pin):
        """
        初始化传感器

        参数:
            name: 传感器名称
            pin: 传感器连接的引脚
        """
        self.name = name
        self.pin = pin
        self.calibration_offset = 0.0
        self.calibration_factor = 1.0

    def calibrate(self, offset=0.0, factor=1.0):
        """
        校准传感器

        参数:
            offset: 偏移量
            factor: 校准因子
        """
        self.calibration_offset = offset
        self.calibration_factor = factor

    def read_raw(self):
        """
        读取原始数据
        子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现read_raw方法")

    def read_calibrated(self):
        """
        读取校准后的数据
        """
        raw_value = self.read_raw()
        return raw_value * self.calibration_factor + self.calibration_offset

class TemperatureSensor(SensorBase):
    """
    温度传感器类 (模拟NTC热敏电阻)
    """
    def __init__(self, pin, resistance=10000, beta=3950):
        """
        初始化温度传感器

        参数:
            pin: ADC引脚
            resistance: 标称电阻(Ω)
            beta: B值常数
        """
        super().__init__("温度传感器", pin)
        self.resistance = resistance
        self.beta = beta
        self.adc = machine.ADC(pin)
        self.adc_reference = 3.3  # 参考电压
        self.series_resistance = 10000  # 串联电阻

    def read_raw(self):
        """
        读取原始ADC值
        """
        return self.adc.read_u16()

    def read_temperature(self):
        """
        读取温度值(摄氏度)
        """
        adc_value = self.read_raw()
        voltage = adc_value * self.adc_reference / 65535

        # 计算热敏电阻阻值
        if voltage > 0:
            thermistor_resistance = (self.adc_reference - voltage) * self.series_resistance / voltage
        else:
            thermistor_resistance = float('inf')

        # 计算温度
        if thermistor_resistance > 0:
            temp_kelvin = 1 / (1/298.15 + (1/self.beta) *
                            (thermistor_resistance / self.resistance - 1))
            temp_celsius = temp_kelvin - 273.15
        else:
            temp_celsius = -999  # 错误值

        return temp_celsius

    def read_calibrated(self):
        """
        读取校准后的温度
        """
        temp = self.read_temperature()
        return temp * self.calibration_factor + self.calibration_offset

class LightSensor(SensorBase):
    """
    光敏传感器类
    """
    def __init__(self, pin):
        """
        初始化光敏传感器

        参数:
            pin: ADC引脚
        """
        super().__init__("光敏传感器", pin)
        self.adc = machine.ADC(pin)

    def read_raw(self):
        """
        读取原始光敏值
        """
        return self.adc.read_u16()

    def read_light_intensity(self):
        """
        读取光照强度 (0-100%)
        """
        raw_value = self.read_raw()
        # 转换为百分比
        intensity = (raw_value / 65535) * 100
        return intensity

    def read_calibrated(self):
        """
        读取校准后的光照强度
        """
        intensity = self.read_light_intensity()
        return intensity * self.calibration_factor + self.calibration_offset

class MotionSensor(SensorBase):
    """
    运动传感器类 (PIR传感器)
    """
    def __init__(self, pin):
        """
        初始化运动传感器

        参数:
            pin: 数字输入引脚
        """
        super().__init__("运动传感器", pin)
        self.pin = machine.Pin(pin, machine.Pin.IN)
        self.last_motion_time = 0

    def read_raw(self):
        """
        读取运动状态
        """
        return self.pin.value()

    def read_motion(self):
        """
        检测运动
        """
        motion = self.read_raw()
        if motion == 1:
            self.last_motion_time = time.time()
        return motion

    def get_motion_time(self):
        """
        获取最后一次运动时间
        """
        return self.last_motion_time

class SoundSensor(SensorBase):
    """
    声音传感器类
    """
    def __init__(self, pin):
        """
        初始化声音传感器

        参数:
            pin: ADC引脚
        """
        super().__init__("声音传感器", pin)
        self.adc = machine.ADC(pin)
        self.sample_count = 10
        self.threshold = 30000

    def read_raw(self):
        """
        读取原始声音值
        """
        samples = []
        for _ in range(self.sample_count):
            samples.append(self.adc.read_u16())
            time.sleep_ms(1)
        return sum(samples) // len(samples)

    def read_sound_level(self):
        """
        读取声音等级
        """
        raw_value = self.read_raw()
        return raw_value

    def detect_clap(self):
        """
        检测拍手声
        """
        current_value = self.read_raw()
        return current_value > self.threshold

class DataLogger:
    """
    数据记录器
    """
    def __init__(self, max_entries=100):
        """
        初始化数据记录器

        参数:
            max_entries: 最大记录条数
        """
        self.max_entries = max_entries
        self.data_buffer = []

    def log_data(self, sensor_name, value, timestamp=None):
        """
        记录数据

        参数:
            sensor_name: 传感器名称
            value: 传感器值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()

        entry = {
            'sensor': sensor_name,
            'value': value,
            'timestamp': timestamp
        }

        self.data_buffer.append(entry)

        # 保持缓冲区大小
        if len(self.data_buffer) > self.max_entries:
            self.data_buffer.pop(0)

    def get_recent_data(self, sensor_name=None, count=10):
        """
        获取最近数据

        参数:
            sensor_name: 传感器名称 (None表示所有传感器)
            count: 返回条数
        """
        if sensor_name is None:
            return self.data_buffer[-count:]
        else:
            filtered = [d for d in self.data_buffer if d['sensor'] == sensor_name]
            return filtered[-count:]

    def get_statistics(self, sensor_name, count=10):
        """
        获取统计信息

        参数:
            sensor_name: 传感器名称
            count: 统计条数
        """
        data = self.get_recent_data(sensor_name, count)
        if not data:
            return None

        values = [d['value'] for d in data]
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1]
        }

    def clear_data(self):
        """
        清空数据
        """
        self.data_buffer.clear()

    def export_json(self):
        """
        导出为JSON格式
        """
        return json.dumps(self.data_buffer, indent=2)

class SensorAlertSystem:
    """
    传感器报警系统
    """
    def __init__(self, led_pin=25, buzzer_pin=15):
        """
        初始化报警系统

        参数:
            led_pin: LED引脚
            buzzer_pin: 蜂鸣器引脚
        """
        self.led = machine.Pin(led_pin, machine.Pin.OUT)
        self.buzzer = machine.Pin(buzzer_pin, machine.Pin.OUT)
        self.alerts = {}

    def add_alert(self, sensor_name, condition, message, callback=None):
        """
        添加报警条件

        参数:
            sensor_name: 传感器名称
            condition: 条件函数 (value) -> bool
            message: 报警消息
            callback: 回调函数
        """
        self.alerts[sensor_name] = {
            'condition': condition,
            'message': message,
            'callback': callback,
            'active': False
        }

    def check_alerts(self, sensor_name, value):
        """
        检查报警条件

        参数:
            sensor_name: 传感器名称
            value: 传感器值
        """
        if sensor_name in self.alerts:
            alert = self.alerts[sensor_name]
            if alert['condition'](value):
                if not alert['active']:
                    self.trigger_alert(sensor_name, alert['message'])
                    alert['active'] = True
                    if alert['callback']:
                        alert['callback'](value)
            else:
                alert['active'] = False

    def trigger_alert(self, sensor_name, message):
        """
        触发报警
        """
        print(f"报警: {sensor_name} - {message}")
        self.visual_alert()
        self.audio_alert()

    def visual_alert(self):
        """
        视觉报警 (LED闪烁)
        """
        for _ in range(3):
            self.led.on()
            time.sleep(0.1)
            self.led.off()
            time.sleep(0.1)

    def audio_alert(self):
        """
        声音报警 (蜂鸣器)
        """
        for _ in range(2):
            self.buzzer.on()
            time.sleep(0.2)
            self.buzzer.off()
            time.sleep(0.1)

async def main():
    """
    主函数 - 演示传感器集成系统
    """
    print("=== 传感器集成系统 ===")

    # 创建传感器
    temp_sensor = TemperatureSensor(26)
    light_sensor = LightSensor(27)
    motion_sensor = MotionSensor(16)
    sound_sensor = SoundSensor(28)

    # 校准传感器
    temp_sensor.calibrate(offset=-2.0, factor=1.05)
    light_sensor.calibrate(factor=0.9)

    # 创建数据记录器
    logger = DataLogger(max_entries=50)

    # 创建报警系统
    alert_system = SensorAlertSystem()

    # 设置报警条件
    alert_system.add_alert(
        "温度传感器",
        lambda x: x > 30,
        "温度过高",
        lambda x: print(f"温度达到 {x:.1f}°C")
    )

    alert_system.add_alert(
        "光敏传感器",
        lambda x: x < 20,
        "环境过暗"
    )

    alert_system.add_alert(
        "运动传感器",
        lambda x: x == 1,
        "检测到运动"
    )

    # 传感器读取任务
    async def read_sensors():
        """传感器读取任务"""
        while True:
            # 读取所有传感器
            temp = temp_sensor.read_calibrated()
            light = light_sensor.read_calibrated()
            motion = motion_sensor.read_motion()
            sound = sound_sensor.read_sound_level()

            # 记录数据
            logger.log_data("温度传感器", temp)
            logger.log_data("光敏传感器", light)
            logger.log_data("运动传感器", motion)
            logger.log_data("声音传感器", sound)

            # 检查报警
            alert_system.check_alerts("温度传感器", temp)
            alert_system.check_alerts("光敏传感器", light)
            alert_system.check_alerts("运动传感器", motion)

            # 打印数据
            print(f"温度: {temp:.1f}°C, 光照: {light:.1f}%, "
                  f"运动: {'是' if motion else '否'}, 声音: {sound}")

            await asyncio.sleep(2)

    # 数据统计任务
    async def show_statistics():
        """数据统计任务"""
        while True:
            await asyncio.sleep(30)  # 每30秒显示一次统计

            print("\n=== 传感器统计 ===")
            temp_stats = logger.get_statistics("温度传感器", 15)
            if temp_stats:
                print(f"温度: 最新{temp_stats['latest']:.1f}°C, "
                      f"平均{temp_stats['avg']:.1f}°C, "
                      f"范围{temp_stats['min']:.1f}-{temp_stats['max']:.1f}°C")

            light_stats = logger.get_statistics("光敏传感器", 15)
            if light_stats:
                print(f"光照: 最新{light_stats['latest']:.1f}%, "
                      f"平均{light_stats['avg']:.1f}%, "
                      f"范围{light_stats['min']:.1f}-{light_stats['max']:.1f}%")

            print("=" * 20)

    # 启动任务
    sensor_task = asyncio.create_task(read_sensors())
    stats_task = asyncio.create_task(show_statistics())

    try:
        await asyncio.gather(sensor_task, stats_task)
    except KeyboardInterrupt:
        print("\n程序停止")

        # 导出数据
        print("\n=== 数据导出 ===")
        print(logger.export_json())

if __name__ == "__main__":
    asyncio.run(main())