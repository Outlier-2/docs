# 嵌入式系统设计 - 第二讲：硬件交互基础

## GPIO编程

### GPIO概述
GPIO（General Purpose Input/Output）是微控制器最基础的硬件接口。

### 数字输出
```python
import machine
import time

# 配置GPIO引脚为输出模式
led_pin = machine.Pin(25, machine.Pin.OUT)

# 输出高低电平
led_pin.value(1)  # 高电平
led_pin.value(0)  # 低电平
```

### 数字输入
```python
# 配置GPIO引脚为输入模式
button_pin = machine.Pin(16, machine.Pin.IN)

# 读取引脚状态
if button_pin.value() == 1:
    print("按钮按下")
else:
    print("按钮释放")
```

### 中断处理
```python
def button_handler(pin):
    print(f"引脚 {pin} 触发中断")

button_pin.irq(trigger=machine.Pin.IRQ_FALLING, handler=button_handler)
```

## 模拟信号处理

### ADC（模数转换器）
```python
import machine

# 配置ADC引脚
adc = machine.ADC(26)  # GPIO26

# 读取模拟值
analog_value = adc.read_u16()
voltage = analog_value * 3.3 / 65535
```

### PWM（脉宽调制）
```python
# 配置PWM
pwm = machine.PWM(machine.Pin(25))
pwm.freq(1000)  # 频率1kHz
pwm.duty_u16(32768)  # 50%占空比
```

## I2C通信

### I2C基础
I2C是一种串行通信协议，适合连接多个设备到同一总线。

### 主机设备
```python
from machine import Pin, I2C

# 初始化I2C
i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=100000)

# 扫描设备
devices = i2c.scan()
print("发现设备:", devices)

# 读取数据
data = i2c.readfrom(0x48, 2)  # 从地址0x48读取2字节

# 写入数据
i2c.writeto(0x48, b'\x01\x02')
```

## SPI通信

### SPI基础
SPI是一种高速同步串行通信协议，适合高速数据传输。

### SPI配置
```python
from machine import Pin, SPI

# 初始化SPI
spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
          sck=Pin(2), mosi=Pin(3), miso=Pin(4))

# 发送数据
spi.write(b'Hello')

# 读取数据
data = spi.read(5)

# 同时收发
data = spi.write_readinto(b'ABC', bytearray(3))
```

## 串口通信

### UART基础
```python
from machine import UART

# 初始化UART
uart = UART(0, baudrate=9600, tx=Pin(0), rx=Pin(1))

# 发送数据
uart.write('Hello World\r\n')

# 接收数据
if uart.any():
    data = uart.read()
    print(data.decode())
```

## 实际应用示例

### 温度传感器读取
```python
import machine
import time
from machine import Pin, I2C

class TemperatureSensor:
    def __init__(self, i2c_addr=0x48):
        self.i2c = I2C(0, scl=Pin(1), sda=Pin(0))
        self.addr = i2c_addr

    def read_temperature(self):
        # 读取温度数据
        data = self.i2c.readfrom(self.addr, 2)
        temp_c = (data[0] << 8 | data[1]) >> 4
        return temp_c * 0.0625

# 使用示例
sensor = TemperatureSensor()
while True:
    temp = sensor.read_temperature()
    print(f"温度: {temp:.1f}°C")
    time.sleep(1)
```

## 实践练习

### 练习1：按钮控制LED
- 使用按钮控制LED开关
- 实现按键消抖

### 练习2：光敏电阻
- 读取光敏电阻值
- 根据光照强度控制LED亮度

### 练习3：I2C设备通信
- 连接I2C传感器
- 实现数据读取和显示

## 课后作业
1. 实现一个完整的传感器监控系统
2. 添加数据记录功能
3. 设计报警阈值功能

## 下一讲预告
实时操作系统概念