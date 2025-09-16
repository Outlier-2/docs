"""
SPI OLED显示屏示例
演示SPI通信和显示屏驱动
"""

import machine
import time
from machine import Pin, SPI

class SPIDisplay:
    """
    SPI OLED显示屏类（模拟SSD1306）
    """
    def __init__(self, width=128, height=64, rst_pin=4, dc_pin=5, cs_pin=6):
        """
        初始化SPI显示屏

        参数:
            width: 显示屏宽度
            height: 显示屏高度
            rst_pin: 复位引脚
            dc_pin: 数据/命令选择引脚
            cs_pin: 片选引脚
        """
        self.width = width
        self.height = height
        self.buffer = bytearray(width * height // 8)

        # 初始化GPIO引脚
        self.rst = Pin(rst_pin, machine.Pin.OUT)
        self.dc = Pin(dc_pin, machine.Pin.OUT)
        self.cs = Pin(cs_pin, machine.Pin.OUT)

        # 初始化SPI
        self.spi = SPI(0, baudrate=1000000, polarity=0, phase=0,
                       sck=Pin(2), mosi=Pin(3), miso=Pin(4))

        # 初始化显示屏
        self.reset()
        self.init_display()

        print(f"SPI显示屏初始化完成，尺寸: {width}x{height}")

    def reset(self):
        """
        复位显示屏
        """
        self.rst.value(1)
        time.sleep_ms(1)
        self.rst.value(0)
        time.sleep_ms(10)
        self.rst.value(1)
        time.sleep_ms(1)

    def init_display(self):
        """
        初始化显示屏命令
        """
        commands = [
            0xAE,  # 关闭显示
            0xD5,  # 设置显示时钟分频
            0x80,
            0xA8,  # 设置多路复用率
            0x3F,
            0xD3,  # 设置显示偏移
            0x00,
            0x40,  # 设置起始行
            0x8D,  # 电荷泵使能
            0x14,
            0x20,  # 设置内存地址模式
            0x00,  # 水平模式
            0xA1,  # 段重映射
            0xC8,  # COM输出方向
            0xDA,  # 设置COM引脚配置
            0x12,
            0x81,  # 设置对比度
            0xCF,
            0xD9,  # 设置预充电周期
            0xF1,
            0xDB,  # 设置VCOMH电压
            0x40,
            0xA4,  # 全局显示开启
            0xA6,  # 设置正常显示
            0xAF,  # 开启显示
        ]

        for cmd in commands:
            self.write_command(cmd)

        self.clear()
        self.show()

    def write_command(self, cmd):
        """
        写入命令
        """
        self.dc.value(0)  # 命令模式
        self.cs.value(0)  # 片选有效
        self.spi.write(bytes([cmd]))
        self.cs.value(1)  # 片选无效

    def write_data(self, data):
        """
        写入数据
        """
        self.dc.value(1)  # 数据模式
        self.cs.value(0)  # 片选有效
        self.spi.write(data)
        self.cs.value(1)  # 片选无效

    def clear(self):
        """
        清空显示缓冲区
        """
        self.buffer = bytearray(len(self.buffer))

    def pixel(self, x, y, color=1):
        """
        设置像素点

        参数:
            x: x坐标
            y: y坐标
            color: 1为点亮，0为熄灭
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        if color:
            self.buffer[x + (y // 8) * self.width] |= 1 << (y % 8)
        else:
            self.buffer[x + (y // 8) * self.width] &= ~(1 << (y % 8))

    def show(self):
        """
        更新显示
        """
        # 设置列地址范围
        self.write_command(0x21)  # 列地址设置
        self.write_command(0)
        self.write_command(self.width - 1)

        # 设置页地址范围
        self.write_command(0x22)  # 页地址设置
        self.write_command(0)
        self.write_command((self.height // 8) - 1)

        # 写入显示数据
        self.write_data(self.buffer)

    def text(self, text, x, y, color=1):
        """
        显示文本（简化版本）
        """
        # 这里使用简化的字符显示
        font = {
            'A': [0x7E, 0x11, 0x11, 0x11, 0x7E],
            'B': [0x7F, 0x49, 0x49, 0x49, 0x36],
            'C': [0x3E, 0x41, 0x41, 0x41, 0x22],
            'D': [0x7F, 0x41, 0x41, 0x22, 0x1C],
            'E': [0x7F, 0x49, 0x49, 0x49, 0x41],
            'F': [0x7F, 0x09, 0x09, 0x09, 0x01],
            'G': [0x3E, 0x41, 0x49, 0x49, 0x7A],
            'H': [0x7F, 0x08, 0x08, 0x08, 0x7F],
            'I': [0x00, 0x41, 0x7F, 0x41, 0x00],
            'J': [0x20, 0x40, 0x41, 0x3F, 0x01],
            'K': [0x7F, 0x08, 0x14, 0x22, 0x41],
            'L': [0x7F, 0x40, 0x40, 0x40, 0x40],
            'M': [0x7F, 0x02, 0x0C, 0x02, 0x7F],
            'N': [0x7F, 0x04, 0x08, 0x10, 0x7F],
            'O': [0x3E, 0x41, 0x41, 0x41, 0x3E],
            'P': [0x7F, 0x09, 0x09, 0x09, 0x06],
            'Q': [0x3E, 0x41, 0x51, 0x21, 0x5E],
            'R': [0x7F, 0x09, 0x19, 0x29, 0x46],
            'S': [0x46, 0x49, 0x49, 0x49, 0x31],
            'T': [0x01, 0x01, 0x7F, 0x01, 0x01],
            'U': [0x3F, 0x40, 0x40, 0x40, 0x3F],
            'V': [0x1F, 0x20, 0x40, 0x20, 0x1F],
            'W': [0x3F, 0x40, 0x38, 0x40, 0x3F],
            'X': [0x63, 0x14, 0x08, 0x14, 0x63],
            'Y': [0x07, 0x08, 0x70, 0x08, 0x07],
            'Z': [0x61, 0x51, 0x49, 0x45, 0x43],
            ' ': [0x00, 0x00, 0x00, 0x00, 0x00],
        }

        for i, char in enumerate(text.upper()):
            if char in font:
                char_data = font[char]
                for col in range(5):
                    for row in range(8):
                        if char_data[col] & (1 << row):
                            self.pixel(x + i * 6 + col, y + row, color)

def main():
    """
    主函数 - 演示SPI显示屏使用
    """
    print("=== SPI显示屏示例 ===")

    try:
        # 初始化显示屏
        display = SPIDisplay()

        # 显示文本
        display.text("HELLO", 10, 10)
        display.text("WORLD", 10, 30)
        display.show()

        print("显示文本: HELLO WORLD")

        # 动画效果
        for i in range(128):
            display.clear()
            display.text("HELLO", 10, 10)
            display.text("WORLD", 10, 30)
            # 绘制移动的像素点
            display.pixel(i, 50, 1)
            display.show()
            time.sleep_ms(50)

        print("动画演示完成")

    except KeyboardInterrupt:
        print("\n程序停止")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()