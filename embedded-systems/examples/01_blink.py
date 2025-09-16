"""
LED闪烁程序 - 嵌入式系统第一个程序
使用MicroPython在Raspberry Pi Pico上实现LED闪烁
"""

import machine
import time

def setup_led():
    """
    初始化LED引脚
    返回: 配置好的LED引脚对象
    """
    # Pico板载LED连接到GPIO25
    led = machine.Pin("LED", machine.Pin.OUT)
    return led

def blink_led(led, on_time=0.5, off_time=0.5):
    """
    LED闪烁函数

    参数:
        led: LED引脚对象
        on_time: LED点亮时间(秒)
        off_time: LED熄灭时间(秒)
    """
    led.on()
    time.sleep(on_time)
    led.off()
    time.sleep(off_time)

def main():
    """
    主函数 - 无限循环闪烁LED
    """
    print("嵌入式系统 - LED闪烁程序")
    print("按Ctrl+C停止程序")

    led = setup_led()

    try:
        while True:
            blink_led(led, 1, 1)  # 1秒亮，1秒灭
    except KeyboardInterrupt:
        print("\n程序停止")
        led.off()

if __name__ == "__main__":
    main()