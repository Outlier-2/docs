"""
通信协议实现示例
演示UART、I2C、SPI等通信协议的实现
"""

import machine
import time
import asyncio
import json
import struct

class UARTCommunication:
    """
    UART通信类
    """
    def __init__(self, uart_id=0, baudrate=9600, tx_pin=0, rx_pin=1):
        """
        初始化UART通信

        参数:
            uart_id: UART编号
            baudrate: 波特率
            tx_pin: 发送引脚
            rx_pin: 接收引脚
        """
        self.uart = machine.UART(uart_id, baudrate=baudrate, tx=tx_pin, rx=rx_pin)
        self.buffer = ""
        self.message_callback = None

    def set_message_callback(self, callback):
        """
        设置消息回调函数
        """
        self.message_callback = callback

    def send_message(self, message):
        """
        发送消息

        参数:
            message: 要发送的消息
        """
        if isinstance(message, str):
            data = message.encode('utf-8')
        else:
            data = message

        self.uart.write(data)
        print(f"发送: {message}")

    def send_json(self, data):
        """
        发送JSON数据
        """
        json_str = json.dumps(data) + "\n"
        self.send_message(json_str)

    def read_message(self):
        """
        读取消息
        """
        if self.uart.any():
            data = self.uart.read()
            if data:
                self.buffer += data.decode('utf-8')

                # 处理完整的行
                while '\n' in self.buffer:
                    line, self.buffer = self.buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        return line
        return None

    def read_json(self):
        """
        读取JSON数据
        """
        message = self.read_message()
        if message:
            try:
                return json.loads(message)
            except json.JSONDecodeError:
                return None
        return None

    async def monitor_uart(self):
        """
        监控UART数据
        """
        while True:
            message = self.read_message()
            if message and self.message_callback:
                self.message_callback(message)
            await asyncio.sleep(0.1)

class I2CDevice:
    """
    I2C设备类
    """
    def __init__(self, i2c_id=0, scl_pin=1, sda_pin=0, freq=100000):
        """
        初始化I2C设备

        参数:
            i2c_id: I2C编号
            scl_pin: 时钟引脚
            sda_pin: 数据引脚
            freq: 频率
        """
        self.i2c = machine.I2C(i2c_id, scl=machine.Pin(scl_pin), sda=machine.Pin(sda_pin), freq=freq)
        self.devices = {}

    def scan_devices(self):
        """
        扫描I2C设备
        """
        devices = self.i2c.scan()
        print(f"发现I2C设备: {[hex(addr) for addr in devices]}")
        return devices

    def write_byte(self, addr, reg, data):
        """
        写入字节

        参数:
            addr: 设备地址
            reg: 寄存器地址
            data: 数据
        """
        self.i2c.writeto_mem(addr, reg, bytes([data]))

    def read_byte(self, addr, reg):
        """
        读取字节

        参数:
            addr: 设备地址
            reg: 寄存器地址
        """
        data = self.i2c.readfrom_mem(addr, reg, 1)
        return data[0]

    def write_bytes(self, addr, reg, data):
        """
        写入多个字节

        参数:
            addr: 设备地址
            reg: 寄存器地址
            data: 数据列表
        """
        self.i2c.writeto_mem(addr, reg, bytes(data))

    def read_bytes(self, addr, reg, length):
        """
        读取多个字节

        参数:
            addr: 设备地址
            reg: 寄存器地址
            length: 读取长度
        """
        data = self.i2c.readfrom_mem(addr, reg, length)
        return list(data)

class I2CTemperatureSensor(I2CDevice):
    """
    I2C温度传感器 (模拟LM75)
    """
    def __init__(self, addr=0x48, **kwargs):
        """
        初始化I2C温度传感器

        参数:
            addr: 传感器地址
        """
        super().__init__(**kwargs)
        self.addr = addr
        self.temp_reg = 0x00
        self.config_reg = 0x01

    def read_temperature(self):
        """
        读取温度值
        """
        # 读取温度寄存器 (2字节)
        data = self.read_bytes(self.addr, self.temp_reg, 2)
        temp_raw = (data[0] << 8 | data[1]) >> 4

        # 转换为摄氏度
        if temp_raw & 0x800:  # 检查符号位
            temp_c = -((~temp_raw + 1) & 0xFFF) * 0.0625
        else:
            temp_c = temp_raw * 0.0625

        return temp_c

    def set_config(self, shutdown=False, os_polarity=0, os_mode=0):
        """
        设置配置寄存器

        参数:
            shutdown: 是否关闭
            os_polarity: OS极性
            os_mode: OS模式
        """
        config = 0
        if shutdown:
            config |= 0x01
        if os_polarity:
            config |= 0x04
        if os_mode:
            config |= 0x02

        self.write_byte(self.addr, self.config_reg, config)

class SPIDevice:
    """
    SPI设备类
    """
    def __init__(self, spi_id=0, baudrate=1000000, polarity=0, phase=0,
                 sck_pin=2, mosi_pin=3, miso_pin=4, cs_pin=5):
        """
        初始化SPI设备

        参数:
            spi_id: SPI编号
            baudrate: 波特率
            polarity: 时钟极性
            phase: 时钟相位
            sck_pin: 时钟引脚
            mosi_pin: MOSI引脚
            miso_pin: MISO引脚
            cs_pin: 片选引脚
        """
        self.spi = machine.SPI(spi_id, baudrate=baudrate, polarity=polarity, phase=phase,
                              sck=machine.Pin(sck_pin), mosi=machine.Pin(mosi_pin),
                              miso=machine.Pin(miso_pin))
        self.cs = machine.Pin(cs_pin, machine.Pin.OUT)
        self.cs.value(1)  # 片选无效

    def write_byte(self, data):
        """
        写入字节

        参数:
            data: 数据
        """
        self.cs.value(0)  # 片选有效
        self.spi.write(bytes([data]))
        self.cs.value(1)  # 片选无效

    def read_byte(self):
        """
        读取字节
        """
        self.cs.value(0)  # 片选有效
        data = self.spi.read(1)
        self.cs.value(1)  # 片选无效
        return data[0]

    def write_read_byte(self, write_data):
        """
        同时写入和读取字节

        参数:
            write_data: 写入的数据
        """
        self.cs.value(0)  # 片选有效
        data = bytearray(1)
        self.spi.write_readinto(bytes([write_data]), data)
        self.cs.value(1)  # 片选无效
        return data[0]

    def write_bytes(self, data):
        """
        写入多个字节

        参数:
            data: 数据列表
        """
        self.cs.value(0)  # 片选有效
        self.spi.write(bytes(data))
        self.cs.value(1)  # 片选无效

    def read_bytes(self, length):
        """
        读取多个字节

        参数:
            length: 读取长度
        """
        self.cs.value(0)  # 片选有效
        data = self.spi.read(length)
        self.cs.value(1)  # 片选无效
        return list(data)

class SPIMemoryDevice(SPIDevice):
    """
    SPI存储器设备 (模拟EEPROM)
    """
    def __init__(self, size=256, **kwargs):
        """
        初始化SPI存储器

        参数:
            size: 存储器大小
        """
        super().__init__(**kwargs)
        self.size = size
        self.memory = bytearray(size)
        self.write_enable = False

        # 命令定义
        self.CMD_WRITE_ENABLE = 0x06
        self.CMD_WRITE_DISABLE = 0x04
        self.CMD_READ_STATUS = 0x05
        self.CMD_WRITE_STATUS = 0x01
        self.CMD_READ_DATA = 0x03
        self.CMD_WRITE_DATA = 0x02

    def enable_write(self):
        """
        使能写入
        """
        self.write_byte(self.CMD_WRITE_ENABLE)
        self.write_enable = True

    def disable_write(self):
        """
        禁止写入
        """
        self.write_byte(self.CMD_WRITE_DISABLE)
        self.write_enable = False

    def read_status(self):
        """
        读取状态寄存器
        """
        self.write_byte(self.CMD_READ_STATUS)
        status = self.read_byte()
        return status

    def write_data(self, addr, data):
        """
        写入数据

        参数:
            addr: 地址
            data: 数据
        """
        if addr + len(data) > self.size:
            raise ValueError("超出存储器范围")

        if not self.write_enable:
            self.enable_write()

        # 发送写入命令
        self.cs.value(0)
        self.spi.write(bytes([self.CMD_WRITE_DATA, addr]))
        self.spi.write(data)
        self.cs.value(1)

        # 模拟写入延迟
        time.sleep_ms(5)

    def read_data(self, addr, length):
        """
        读取数据

        参数:
            addr: 地址
            length: 读取长度
        """
        if addr + length > self.size:
            raise ValueError("超出存储器范围")

        # 发送读取命令
        self.cs.value(0)
        self.spi.write(bytes([self.CMD_READ_DATA, addr]))
        data = self.spi.read(length)
        self.cs.value(1)

        return data

class CommunicationManager:
    """
    通信管理器
    """
    def __init__(self):
        """
        初始化通信管理器
        """
        self.uart_comm = UARTCommunication()
        self.i2c_device = I2CDevice()
        self.spi_device = SPIDevice()
        self.message_handlers = {}

    def register_handler(self, protocol, handler):
        """
        注册消息处理器

        参数:
            protocol: 协议名称 ('uart', 'i2c', 'spi')
            handler: 处理器函数
        """
        self.message_handlers[protocol] = handler

    def handle_uart_message(self, message):
        """
        处理UART消息
        """
        print(f"UART收到: {message}")

        # 解析JSON消息
        try:
            data = json.loads(message)
            self.process_command(data)
        except json.JSONDecodeError:
            pass

    def process_command(self, command):
        """
        处理命令

        参数:
            command: 命令字典
        """
        cmd_type = command.get('type')
        cmd_data = command.get('data', {})

        if cmd_type == 'i2c_scan':
            devices = self.i2c_device.scan_devices()
            response = {'type': 'i2c_scan_result', 'devices': devices}
            self.uart_comm.send_json(response)

        elif cmd_type == 'i2c_read':
            addr = cmd_data.get('addr')
            reg = cmd_data.get('reg')
            if addr and reg is not None:
                value = self.i2c_device.read_byte(addr, reg)
                response = {'type': 'i2c_read_result', 'addr': addr, 'reg': reg, 'value': value}
                self.uart_comm.send_json(response)

        elif cmd_type == 'i2c_write':
            addr = cmd_data.get('addr')
            reg = cmd_data.get('reg')
            value = cmd_data.get('value')
            if addr and reg is not None and value is not None:
                self.i2c_device.write_byte(addr, reg, value)
                response = {'type': 'i2c_write_result', 'success': True}
                self.uart_comm.send_json(response)

    async def run(self):
        """
        运行通信管理器
        """
        # 设置UART消息回调
        self.uart_comm.set_message_callback(self.handle_uart_message)

        # 启动UART监控
        uart_task = asyncio.create_task(self.uart_comm.monitor_uart())

        # 发送欢迎消息
        self.uart_comm.send_json({
            'type': 'system_ready',
            'message': '通信系统已就绪'
        })

        # 运行主循环
        try:
            await asyncio.gather(uart_task)
        except KeyboardInterrupt:
            print("通信管理器停止")

async def main():
    """
    主函数 - 演示通信协议实现
    """
    print("=== 通信协议实现示例 ===")

    # 创建通信管理器
    comm_manager = CommunicationManager()

    # 启动通信系统
    await comm_manager.run()

if __name__ == "__main__":
    asyncio.run(main())