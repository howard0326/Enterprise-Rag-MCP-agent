# RAG + MCP 智能助手评测报告

- 执行时间：2026-09-22T19:11:49+08:00 → 2026-09-22T19:12:47+08:00
- 用例总数：**20**
- 意图识别路径：**LLM（deepseek-chat）**

## 1. 总体指标

| 指标 | 数值 | 通过的用例数 |
| --- | --- | --- |
| 意图准确率 | **100.0%** | 20/20 |
| 工具选择准确率 | **100.0%** | 20/20 |
| 回答包含率 | **100.0%** | 20/20 |
| 三项全通过 | **100.0%** | 20/20 |
| 端到端延迟（均值） | **2.13s** | P50 2.22s / P95 2.73s / max 3.00s |

## 2. 分类指标

| 分类 | 用例数 | 意图准确率 | 工具选择准确率 | 回答包含率 | 全通过 | 平均延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| action_request | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 1.75s |
| data_query | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 1.73s |
| knowledge_qa | 8 | 100.0% | 100.0% | 100.0% | 100.0% | 2.36s |
| mixed | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 2.68s |

## 3. 失败类型分布

无失败用例。

## 4. 逐条结果

| ID | 分类 | 输入 | 期望意图 | 实际意图 | 期望工具 | 实际工具 | 回答命中 | 延迟 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KB-01 | knowledge_qa | 退货政策是什么？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 7 个自然日/无理由退货/签收 | 2.5s | ✅ |
| KB-02 | knowledge_qa | 无理由退货的期限是多久？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 7 个自然日/15 个自然日 | 2.3s | ✅ |
| KB-03 | knowledge_qa | 退款一般多久到账？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 3~5 个工作日/质检/工作日 | 2.2s | ✅ |
| KB-04 | knowledge_qa | 工单优先级 urgent 的响应时限是多久？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 15 分钟/urgent/4 小时 | 2.2s | ✅ |
| KB-05 | knowledge_qa | 创建工单必须填写哪些信息？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 标题/描述/优先级 | 2.4s | ✅ |
| KB-06 | knowledge_qa | 哪些商品不支持退货？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 定制/数字/拆封 | 2.2s | ✅ |
| KB-07 | knowledge_qa | 已签收的订单要退货应该走什么流程？ | knowledge_qa | knowledge_qa | search_documents | search_documents | 退货申请/delivered/签收 | 2.7s | ✅ |
| KB-08 | knowledge_qa | 退款率是怎么计算的？ | knowledge_qa | knowledge_qa | search_documents | search_documents | refunded/退款率/订单 | 2.2s | ✅ |
| DB-01 | data_query | 查一下 CUST-1001 的所有订单。 | data_query | data_query | query_orders | query_orders | ORD-001/ORD-002/ORD-007 | 2.2s | ✅ |
| DB-02 | data_query | 订单 ORD-001 现在是什么状态？ | data_query | data_query | query_orders | query_orders | delivered/ORD-001/签收 | 1.8s | ✅ |
| DB-03 | data_query | ORD-001 的订单金额是多少？ | data_query | data_query | query_orders | query_orders | 299/CNY | 1.3s | ✅ |
| DB-04 | data_query | CUST-1002 有哪些订单？ | data_query | data_query | query_orders | query_orders | ORD-003/ORD-004 | 1.6s | ✅ |
| DB-05 | data_query | CUST-1001 在 2025 年 3 月的订单有 | data_query | data_query | query_orders | query_orders | ORD-001/ORD-002 | 1.8s | ✅ |
| DB-06 | data_query | 哪些订单已经退款了？ | data_query | data_query | query_orders | query_orders | refunded | 1.7s | ✅ |
| MIX-01 | mixed | 订单 ORD-001 符合退货政策吗？ | data_query,knowledge_qa | data_query,knowledge_qa | query_orders,search_documents | query_orders,search_documents | delivered/签收 | 3.0s | ✅ |
| MIX-02 | mixed | ORD-001 已经签收了，我还能退货吗？需要满足什 | data_query,knowledge_qa | knowledge_qa,data_query | query_orders,search_documents | query_orders,search_documents | 退货申请/包装/发票 | 2.6s | ✅ |
| MIX-03 | mixed | CUST-1003 的订单里有没有在退货期限内的？ | data_query,knowledge_qa | data_query,knowledge_qa | query_orders,search_documents | query_orders,search_documents | CUST-1003/ORD-005/个自然日 | 2.4s | ✅ |
| ACT-01 | action_request | 帮我给订单 ORD-001 创建一个高优先级工单，退 | action_request | action_request,data_query,knowledge_qa | create_ticket,query_orders | create_ticket,query_orders | TKT-/high/ORD-001 | 2.1s | ✅ |
| ACT-02 | action_request | 客户反馈 ORD-004 一直没发货，麻烦建个紧急工 | action_request | action_request,data_query | create_ticket,query_orders | create_ticket,query_orders | TKT-/urgent/ORD-004 | 1.9s | ✅ |
| ACT-03 | action_request | 请为 CUST-1002 的订单问题登记一个工单。 | action_request | action_request,data_query,knowledge_qa | create_ticket | create_ticket | TKT-/CUST-1002 | 1.2s | ✅ |

## 5. 失败案例分析

全部用例通过，无失败案例。

## 6. 结论与改进方向

- 意图准确率 100.0%、工具选择准确率 100.0%、回答包含率 100.0%。
