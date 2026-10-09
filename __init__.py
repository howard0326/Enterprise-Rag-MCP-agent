"""企业级 RAG + MCP 智能助手。

模块划分：
    config/         全局配置加载（settings.yaml + .env）
    rag_pipeline/   文档解析、向量索引、混合检索、增量摄取 CLI
    mcp_servers/    MCP 工具服务器（knowledge_base / business_db）与注册表
    backend/        LangGraph Agent 编排与 FastAPI 服务
    eval/           评测用例、评测执行器与可观测性报告
"""

__version__ = "0.1.0"
