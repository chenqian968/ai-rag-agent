```
# 大数据知识库（RAG 测试专用）

## 1. 大数据定义与 4V
大数据一般指无法在可承受时间内用常规工具进行捕捉、存储、管理、分析的海量数据集。经典特征为 4V：
- **Volume（大量）**：数据规模巨大，TB、PB、EB 级别；
- **Velocity（高速）**：数据产生、流入速度快，常需要实时处理；
- **Variety（多样）**：结构化、半结构化、非结构化数据并存（数据库、日志、图片、文本、音视频）；
- **Value（价值密度低）**：海量数据里有效信息稀疏，需要挖掘才能提取价值。

部分教材会增加第 5V：Veracity（真实性），代表数据可信程度、噪声、脏数据问题。

## 2. Hadoop 生态
Hadoop 是开源分布式大数据基础框架，核心三大组件：
1. **HDFS（Hadoop Distributed File System）**：分布式文件存储，把文件切分成块（默认 128MB），多副本保存，适合存超大文件；主从架构：NameNode（元数据）、DataNode（真实数据块）、SecondaryNameNode（辅助检查点，不是备份）。
2. **MapReduce**：第一代分布式计算引擎，分为 Map（分片处理）、Shuffle（混洗排序）、Reduce（聚合输出），磁盘 IO 多，适合离线批处理。
3. **YARN（Yet Another Resource Negotiator）**：资源调度管理器，负责集群 CPU、内存分配；包含 ResourceManager、NodeManager、ApplicationMaster。

周边生态：
- **Hive**：数据仓库，把 SQL 翻译成 MapReduce / Spark，用类 SQL（HiveQL）查询；
- **HBase**：分布式列式 NoSQL，适合随机实时读写；
- **ZooKeeper**：分布式协调服务，用来选主、配置同步、分布式锁；
- **Sqoop**：在关系型数据库（MySQL、Oracle）和 HDFS 之间导入导出；
- **Flume**：日志采集工具，流式收集日志写入 HDFS / Kafka；
- **Oozie / Azkaban**：工作流调度。

## 3. Spark
Apache Spark 是内存分布式计算框架，比 MapReduce 快很多，核心抽象：RDD、DataFrame、DataSet。
- **RDD（弹性分布式数据集）**：不可变、可分区、支持容错；转换算子（lazy）、行动算子（触发计算）；
- **DataFrame**：带 schema，结构化，优化更好；
- **DataSet**：强类型，兼顾类型安全与优化。

四大模块：
- Spark Core：基础 API、任务调度；
- Spark SQL：结构化查询；
- Spark Streaming / Structured Streaming：流处理；
- MLlib：机器学习库；
- GraphX：图计算。

## 4. Kafka
Kafka：分布式消息队列 / 流平台。
- 核心概念：Producer（生产者）、Consumer（消费者）、Broker、Topic、Partition、Offset；
- 分区保证并行消费；offset 记录读到哪里；
- 常用于数据解耦、削峰填谷、实时数据管道；
- 高吞吐，适合日志、埋点、实时数据流。

## 5. 流计算 vs 批计算
- **批计算（Batch）**：数据已经全部落盘，一次性处理。代表：MapReduce、Hive、Spark Batch；延迟高，适合报表、离线统计。
- **流计算（Stream）**：数据源源不断进来，持续处理。代表：Flink、Spark Structured Streaming；低延迟，适合实时大屏、告警、实时推荐。

Apache Flink：天然流优先，支持事件时间、水位线（Watermark）、状态管理、精确一次（exactly-once）语义，现在工业界实时场景非常常用。

## 6. 数据仓库、数据湖、湖仓一体
- **数据仓库（Data Warehouse）**：结构化为主，面向分析，分层建模（ODS、DWD、DWS、ADS）；
- **数据湖（Data Lake）**：原始数据直接存放，可以存任意格式；低成本，但是治理难；
- **湖仓一体（Lakehouse）**：结合两者，底层像湖存原始数据，上层提供仓库的 ACID、查询能力；典型项目：Iceberg、Hudi、Delta Lake。

分层名词：
- ODS：原始层，几乎原样接入；
- DWD：明细层，清洗、过滤脏数据；
- DWS：汇总层，按维度聚合；
- ADS：应用层，给报表、业务使用。

## 7. 分布式基础概念
- **副本（Replication）**：多份拷贝，保证节点挂掉不丢数据；
- **分片 / 分区（Shard / Partition）**：把数据切开分散到多台机器；
- **CAP 理论**：分布式系统无法同时满足一致性、可用性、分区容错；大数据系统大多选择 AP 或者 CP；
- **BASE**：基本可用、软状态、最终一致性，是很多 NoSQL 的设计思路；
- **一致性哈希**：用来减少扩容缩容时大量数据迁移。

## 8. ETL & ELT
- **ETL**：Extract → Transform → Load；先抽取、转换清洗，再入库；传统数仓常用；
- **ELT**：Extract → Load → Transform；先把原始数据丢进湖，后面再按需转换；大数据湖常用。

## 9. 常见面试 / 测试问答
> Q：NameNode 挂了怎么办？
> A：HDFS 默认没有热备 NameNode，生产一般使用 HA（高可用）搭配 ZooKeeper 实现主备切换。

> Q：MapReduce Shuffle 是做什么？
> A：把 Map 的输出按键分组、排序，发送给对应的 Reduce。

> Q：Spark 为什么比 MapReduce 快？
> A：中间结果优先放内存，减少磁盘读写；DAG 调度，减少任务启动开销。

> Q：Kafka offset 存在哪里？
> A：新版本默认存在 Kafka 内部 topic（__consumer_offsets），旧版本存在 ZooKeeper。

> Q：事件时间、处理时间？
> A：事件时间是数据本身发生的时间；处理时间是服务器收到这条数据的时间。

## 10. 大数据安全与治理
包含数据脱敏、权限控制、数据血缘、元数据管理、数据质量校验；工具例如 Apache Atlas 做元数据和血缘。

## 11. 典型项目链路
埋点日志 → Flume / Filebeat → Kafka → Flink / Spark Streaming → HDFS / Hudi → Hive / Presto → 报表、实时看板、推荐系统。
```