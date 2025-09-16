# 嵌入式系统设计课程

基于CS50和密歇根大学EECS373课程理念的嵌入式系统设计教程，使用Python语言实现。

## 课程特色

### 🎯 学习目标
- 掌握嵌入式系统基础概念
- 学习Python在嵌入式系统中的应用
- 理解硬件交互和通信协议
- 掌握实时操作系统原理
- 完成实际项目开发

### 📚 课程结构
1. **嵌入式系统基础** - 介绍和开发环境
2. **硬件交互** - GPIO、I2C、SPI编程
3. **实时系统** - RTOS概念和多任务编程
4. **传感器接口** - 多种传感器数据采集
5. **通信协议** - UART、I2C、SPI实现
6. **项目实践** - 智能家居和环境监测

### 🛠️ 开发环境
- **硬件**: Raspberry Pi Pico / ESP32
- **软件**: MicroPython / Thonny IDE
- **语言**: Python 3
- **协议**: I2C、SPI、UART

## 课程内容

### 讲座 (Lectures)
- `lecture1-intro.md` - 嵌入式系统介绍
- `lecture2-hardware-interaction.md` - 硬件交互基础
- `lecture3-rtos-concepts.md` - 实时操作系统概念

### 示例代码 (Examples)
- `01_blink.py` - LED闪烁程序
- `02_gpio_button.py` - GPIO按钮控制
- `03_i2c_sensor.py` - I2C温度传感器
- `04_spi_display.py` - SPI显示屏驱动
- `05_rtos_tasks.py` - RTOS任务管理
- `06_sensor_integration.py` - 传感器集成系统
- `07_communication_protocols.py` - 通信协议实现

### 项目 (Projects)
- `smart-home-system.py` - 智能家居系统
- `environment-monitoring.py` - 环境监测站

### 实验 (Labs)
- `lab1-environment-setup.md` - 开发环境搭建

## 快速开始

### 1. 安装开发环境
```bash
# 下载Thonny IDE
# 访问 https://thonny.org/ 下载并安装
```

### 2. 硬件准备
- Raspberry Pi Pico 或 ESP32 开发板
- USB 数据线
- 面包板和杜邦线
- 基础电子元件 (LED、电阻、按钮等)

### 3. 第一个程序
```python
# 运行 examples/01_blink.py
import machine
import time

led = machine.Pin("LED", machine.Pin.OUT)

while True:
    led.on()
    time.sleep(1)
    led.off()
    time.sleep(1)
```

## 学习路径

### 入门阶段
1. 了解嵌入式系统概念
2. 搭建开发环境
3. 完成LED闪烁实验
4. 学习GPIO编程

### 进阶阶段
1. 掌握I2C、SPI通信
2. 学习传感器接口
3. 理解RTOS概念
4. 实现多任务编程

### 项目阶段
1. 完成智能家居系统
2. 开发环境监测站
3. 集成多种传感器
4. 实现数据记录和分析

## 核心概念

### 🔌 硬件交互
- **GPIO**: 数字输入输出
- **ADC**: 模数转换
- **PWM**: 脉宽调制
- **中断**: 事件驱动编程

### 📡 通信协议
- **UART**: 串口通信
- **I2C**: 双线串行总线
- **SPI**: 四线高速接口

### ⏱️ 实时系统
- **任务调度**: 优先级和轮转
- **同步机制**: 信号量和队列
- **内存管理**: 动态分配和优化

### 🌐 网络连接
- **WiFi**: 无线网络接入
- **MQTT**: 物联网协议
- **HTTP**: Web服务接口

## 项目特点

### 智能家居系统
- 温湿度监控
- 自动化控制
- 远程操作
- 报警功能

### 环境监测站
- 多传感器集成
- 数据记录和分析
- 实时报警
- 趋势分析

## 最佳实践

### 代码风格
- 使用中文注释
- 模块化设计
- 错误处理
- 资源管理

### 系统设计
- 状态机模式
- 事件驱动架构
- 分层设计
- 配置管理

### 调试技巧
- 串口调试
- LED指示
- 日志记录
- 性能优化

## 扩展学习

### 相关技术
- **MicroPython**: Python在微控制器上的实现
- **FreeRTOS**: 专业RTOS系统
- **MQTT**: 物联网通信协议
- **Web技术**: RESTful API和WebSocket

### 进阶项目
- 智能农业系统
- 工业监控设备
- 智能家居网关
- 健康监测设备

## 常见问题

### Q: 为什么选择Python而不是C/C++？
A: Python开发效率高，学习成本低，适合快速原型开发和教育用途。

### Q: 如何处理实时性要求高的场景？
A: 可以结合MicroPython的异步特性和中断机制，或者使用专门的RTOS。

### Q: 如何选择合适的开发板？
A: Raspberry Pi Pico适合入门，ESP32提供WiFi和蓝牙功能，STM32适合工业应用。

## 贡献指南

欢迎提交问题和改进建议：
1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- 哈佛大学CS50课程
- 密歇根大学EECS373课程
- MicroPython社区
- 所有贡献者和使用者

---

**开始你的嵌入式系统学习之旅吧！** 🚀