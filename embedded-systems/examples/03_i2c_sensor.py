"""
I2C温度传感器示例
演示I2C通信和传感器数据读取
"""

import machine
import time
from machine import Pin, I2C

class I2CTemperatureSensor:
    """
    I2C温度传感器类（模拟LM75）
    """
    def __init__(self, i2c_addr=0x48, scl_pin=1, sda_pin=0):
        """
        初始化I2C温度传感器

        参数:
            i2c_addr: I2C设备地址
            scl_pin: I2C时钟引脚
            sda_pin: I2C数据引脚
        """
        self.addr = i2c_addr
        self.i2c = I2C(0, scl=Pin(scl_pin), sda=Pin(sda_pin), freq=100000)

        # 检查设备是否存在
        if self.addr not in self.i2c.scan():
            raise ValueError(f"未找到I2C设备，地址: 0x{self.addr:02x}")

        print(f"I2C温度传感器初始化完成，地址: 0x{self.addr:02x}")

    def read_temperature(self):
        """
        读取温度值

        返回:
            温度值（摄氏度）
        """
        try:
            # 读取温度寄存器（2字节）
            data = self.i2c.readfrom(self.addr, 2)

            # 转换为温度值
            temp_raw = (data[0] << 8 | data[1]) >> 4
            temperature = temp_raw * 0.0625

            return temperature
        except Exception as e:
            print(f"读取温度失败: {e}")
            return None

    def read_config(self):
        """
        读取配置寄存器

        返回:
            配置寄存器值
        """
        try:
            # 发送配置寄存器地址
            self.i2c.writeto(self.addr, b'\x01')
            # 读取配置值
            config = self.i2c.readfrom(self.addr, 1)
            return config[0]
        except Exception as e:
            print(f"读取配置失败: {e}")
            return None

class TemperatureMonitor:
    """
    温度监控类
    """
    def __init__(self, sensor, high_threshold=25.0, low_threshold=20.0):
        """
        初始化温度监控器

        参数:
            sensor: 温度传感器对象
            high_threshold: 高温阈值
            low_threshold: 低温阈值
        """
        self.sensor = sensor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.led = machine.Pin(25, machine.Pin.OUT)
        self.alarm_led = machine.Pin(16, machine.Pin.OUT)

        print(f"温度监控器初始化完成")
        print(f"高温阈值: {high_threshold}°C")
        print(f"低温阈值: {low_threshold}°C")

    def check_temperature(self):
        """
        检查温度并触发报警
        """
        temp = self.sensor.read_temperature()
        if temp is None:
            return False

        print(f"当前温度: {temp:.1f}°C")

        # 温度报警逻辑
        if temp > self.high_threshold:
            print("警告: 温度过高!")
            self.alarm_led.on()
            self.led.on()  # 快速闪烁
            time.sleep(0.1)
            self.led.off()
        elif temp < self.low_threshold:
            print("警告: 温度过低!")
            self.alarm_led.on()
            self.led.on()  # 慢速闪烁
            time.sleep(0.5)
            self.led.off()
        else:
            self.alarm_led.off()
            self.led.off()  # 正常温度

        return True

def main():
    """
    主函数 - 演示I2C温度传感器使用
    """
    print("=== I2C温度传感器示例 ===")

    try:
        # 初始化传感器
        sensor = I2CTemperatureSensor()

        # 创建监控器
        monitor = TemperatureMonitor(sensor, 25.0, 20.0)

        print("开始温度监控...")
        print("按Ctrl+C停止")

        while True:
            monitor.check_temperature()
            time.sleep(2)  # 每2秒读取一次

    except KeyboardInterrupt:
        print("\n程序停止")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()