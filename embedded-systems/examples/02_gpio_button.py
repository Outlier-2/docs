"""
GPIO按钮控制LED示例
演示数字输入输出和中断处理
"""

import machine
import time

class ButtonLED:
    """
    按钮控制LED类
    """
    def __init__(self, led_pin=25, button_pin=16):
        """
        初始化按钮和LED

        参数:
            led_pin: LED引脚号
            button_pin: 按钮引脚号
        """
        self.led = machine.Pin(led_pin, machine.Pin.OUT)
        self.button = machine.Pin(button_pin, machine.Pin.IN, machine.Pin.PULL_UP)
        self.led_state = False

        # 设置按钮中断
        self.button.irq(trigger=machine.Pin.IRQ_FALLING, handler=self.button_isr)

        print(f"按钮LED系统初始化完成")
        print(f"LED引脚: {led_pin}, 按钮引脚: {button_pin}")

    def button_isr(self, pin):
        """
        按钮中断服务程序
        """
        # 简单的消抖处理
        time.sleep_ms(50)
        if self.button.value() == 0:  # 再次确认按钮按下
            self.led_state = not self.led_state
            self.led.value(self.led_state)
            print(f"LED状态: {'ON' if self.led_state else 'OFF'}")

    def toggle_led(self):
        """
        切换LED状态
        """
        self.led_state = not self.led_state
        self.led.value(self.led_state)
        return self.led_state

    def set_led(self, state):
        """
        设置LED状态

        参数:
            state: True为点亮，False为熄灭
        """
        self.led_state = state
        self.led.value(1 if state else 0)
        return self.led_state

def main():
    """
    主函数 - 演示按钮控制LED
    """
    print("=== 按钮LED控制示例 ===")
    print("按下GPIO16按钮切换LED状态")

    button_led = ButtonLED()

    try:
        while True:
            # 主循环可以执行其他任务
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n程序停止")
        button_led.set_led(False)

if __name__ == "__main__":
    main()