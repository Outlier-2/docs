# 分布式系统设计课程

基于MIT 6.824课程的完整分布式系统设计教程，包含23个讲义，涵盖从基础概念到高级实现的全方位内容。

## 课程结构

### 📚 课程概览
- [课程总览](./overview.mdx) - 课程介绍、学习目标、先修要求

### 🏗️ 基础部分 (Lecture 1-4)
1. [Lecture 1: 介绍](./lecture-01.mdx) - 分布式系统基础概念和MapReduce
2. [Lecture 2: RPC和线程](./lecture-02.mdx) - 远程过程调用和Go并发编程
3. [Lecture 3: 主从复制](./lecture-03.mdx) - 基本的容错机制
4. [Lecture 4: 一致性与线性化](./lecture-04.mdx) - 分布式一致性模型

### 🔍 共识算法 (Lecture 5-7)
5. [Lecture 5: 容错机制 - Raft (1)](./lecture-05.mdx) - 领导者选举
6. [Lecture 6: Go模式](./lecture-06.mdx) - 分布式系统中的Go编程实践
7. [Lecture 7: 容错机制 - Raft (2)](./lecture-07.mdx) - 日志复制和一致性

### 💾 分布式存储 (Lecture 8-9)
8. [Lecture 8: GFS](./lecture-08.mdx) - Google文件系统设计
9. [Lecture 9: ZooKeeper](./lecture-09.mdx) - 分布式协调服务

### 🔄 事务与一致性 (Lecture 10-14)
10. [Lecture 10: 分布式事务](./lecture-10.mdx) - 两阶段提交和三阶段提交
11. [Lecture 11: Lab 3A+B 问答](./lecture-11.mdx) - Raft实现问题解答
12. [Lecture 12: Spanner](./lecture-12.mdx) - 全球分布式数据库
13. [Lecture 13: 乐观并发控制](./lecture-13.mdx) - 高性能并发控制机制
14. [Lecture 14: Chardonnay](./lecture-14.mdx) - 现代分布式数据库

### 🚀 高级主题 (Lecture 15-23)
15. [Lecture 15: 分布式系统验证](./lecture-15.mdx) - 形式化验证技术
16. [Lecture 16: 缓存系统](./lecture-16.mdx) - 分布式缓存策略
17. [Lecture 17: 消息队列](./lecture-17.mdx) - 异步通信模式
18. [Lecture 18: 负载均衡](./lecture-18.mdx) - 服务发现和流量分配
19. [Lecture 19: 监控和调试](./lecture-19.mdx) - 分布式系统可观测性
20. [Lecture 20: 安全性](./lecture-20.mdx) - 分布式系统安全机制
21. [Lecture 21: 容器化和编排](./lecture-21.mdx) - Docker和Kubernetes
22. [Lecture 22: 服务网格](./lecture-22.mdx) - 微服务架构管理
23. [Lecture 23: 未来趋势](./lecture-23.mdx) - 新兴技术和发展方向

## 📝 课程特色

### 🎯 严谨的学术基础
- 基于MIT 6.824课程体系
- 结合最新研究成果
- 理论与实践并重

### 🔧 实用的工程实践
- 真实的系统案例分析
- 可重用的代码示例
- 最佳实践指导

### 🛠️ 现代化的技术栈
- Go语言实现
- 容器化部署
- 云原生架构

### 📚 详细的文档格式
- 参考CS50的MDX格式
- 包含视频链接、代码示例、练习题
- 结构化的学习路径

## 🎓 学习路径

### 🌟 基础学习 (Lecture 1-7)
1. 理解分布式系统基本概念
2. 掌握RPC和并发编程
3. 学习主从复制和Raft共识算法

### 🔄 进阶学习 (Lecture 8-14)
1. 深入分布式存储系统
2. 掌握分布式事务处理
3. 学习现代数据库设计

### 🚀 高级学习 (Lecture 15-23)
1. 了解系统验证和缓存技术
2. 掌握微服务和容器化
3. 关注未来技术趋势

## 🛠️ 实践项目

课程包含多个实践项目：
- Lab 1: MapReduce实现
- Lab 2: 主从复制系统
- Lab 3: Raft共识算法
- Lab 4: 分布式事务
- Lab 5: 分布式系统验证
- 最终项目: 完整的分布式系统设计

## 📖 学习建议

1. **循序渐进**：按照讲义顺序学习，每个讲义都基于前面的内容
2. **理论结合实践**：每个概念都要配合代码实现
3. **动手实验**：在本地环境中部署和测试分布式系统
4. **阅读论文**：深入理解经典论文的设计思想
5. **参与社区**：加入分布式系统的技术讨论

## 🚀 开始学习

1. 从[课程概览](./overview.mdx)开始了解整体框架
2. 按照讲义顺序学习，从[Lecture 1](./lecture-01.mdx)开始
3. 完成每个讲义的练习题和实践项目
4. 参考扩展资源深入学习

## 📄 版权说明

本课程基于MIT 6.824 Distributed Systems课程，遵循MIT开源许可证。

---

*欢迎来到分布式系统设计的精彩世界！让我们一起探索构建大规模、高可用系统的艺术与科学。*