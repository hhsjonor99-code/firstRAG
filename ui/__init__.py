"""firstRAG Streamlit UI 子包。

不存放业务逻辑；仅负责：

- :mod:`ui.state` —— ``st.session_state`` 初始化与辅助函数
- :mod:`ui.service_factory` —— RAG 服务实例化（``st.cache_resource``）
- :mod:`ui.components` —— 纯函数 UI 辅助（位置格式化 / 文件大小 / 错误映射 / 引用渲染）
"""
