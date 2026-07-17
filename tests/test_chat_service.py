"""ChatService 单元测试。

所有测试：

- 使用 ``FakeRetriever`` 与 ``FakeLLMClient``，**不**调用真实 MiniMax / SiliconFlow；
- 使用临时目录（若需要）；
- 不记录或输出 API Key、Prompt、文档正文。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import Settings  # noqa: E402

from rag.chat_service import (  # noqa: E402
    ChatGenerationError,
    ChatService,
    ChatServiceError,
    NO_EVIDENCE_REPLY,
    QueryRewriteError,
    QueryValidationError,
)
from rag.llm_client import LLMError  # noqa: E402
from rag.models import (  # noqa: E402
    CHAT_EVENT_DONE,
    CHAT_EVENT_ERROR,
    CHAT_EVENT_REWRITE,
    CHAT_EVENT_SOURCES,
    CHAT_EVENT_TOKEN,
    ChatMessage,
    DocumentChunk,
    RetrievedChunk,
)
from rag.prompt_builder import PromptBuilder  # noqa: E402
from rag.retriever import RetrieverError  # noqa: E402

from tests._fakes import (  # noqa: E402
    FakeLLMClient,
    FakeRetriever,
    StreamingFakeLLMClient,
    make_chunk,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _new_settings(**overrides) -> Settings:
    with mock.patch.dict("os.environ", {}, clear=True):
        return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@pytest.fixture
def settings() -> Settings:
    return _new_settings(retrieval_top_k=3, max_history_turns=4)


def _build_retrieved(
    items: list[tuple[str, str, str, float, str]],
) -> list[RetrievedChunk]:
    """items: [(chunk_id, doc_id, content, score, citation_id)]"""
    out = []
    for cid, did, content, score, cit in items:
        chunk = DocumentChunk(
            chunk_id=cid,
            document_id=did,
            content=content,
            source_name="doc.txt",
            chunk_index=0,
        )
        out.append(RetrievedChunk(chunk=chunk, score=score, citation_id=cit))
    return out


def _make_history(*pairs: tuple[str, str]) -> list[ChatMessage]:
    """构造多轮历史：[(user, assistant), ...]"""
    out: list[ChatMessage] = []
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, (u, a) in enumerate(pairs):
        out.append(ChatMessage(
            role="user", content=u, created_at=base.replace(minute=i*2)
        ))
        out.append(ChatMessage(
            role="assistant", content=a, citations=[],
            created_at=base.replace(minute=i*2+1),
        ))
    return out


@pytest.fixture
def sample_retrieved() -> list[RetrievedChunk]:
    return _build_retrieved([
        ("c1", "d1", "高血压是一种常见慢性病。", 0.95, "S1"),
        ("c2", "d1", "高血压需要定期监测血压。", 0.88, "S2"),
    ])


# ---------------------------------------------------------------------------
# 26. 无历史单轮问答
# ---------------------------------------------------------------------------
def test_ask_no_history_single_turn(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="高血压是一种慢性病 [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("什么是高血压？")
    assert msg.role == "assistant"
    assert "高血压" in msg.content
    # citations 只来自答案中实际引用的 [S1]
    assert len(msg.citations) == 1
    assert {c.citation_id for c in msg.citations} == {"S1"}
    # metadata 仍记录候选
    assert msg.metadata.get("retrieval_count") == 2
    assert msg.metadata.get("standalone_query") == "什么是高血压？"
    assert msg.metadata.get("candidate_citation_ids") == ["S1", "S2"]
    assert msg.metadata.get("used_citation_ids") == ["S1"]
    # 检索被调用一次
    assert retriever.call_count == 1


# ---------------------------------------------------------------------------
# 27. 无历史不调用改写 LLM
# ---------------------------------------------------------------------------
def test_no_history_no_rewrite_call(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="答案 X [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    svc.ask("Q1")
    # 改写阶段未触发，complete 只应被调用 1 次（回答阶段）
    assert llm.call_count == 1


# ---------------------------------------------------------------------------
# 28. 有历史问题改写
# ---------------------------------------------------------------------------
def test_rewrite_with_history(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    # 第一次调用（改写）返回 "高血压 治疗 药物"
    # 第二次调用（回答）返回 "答案"
    llm = FakeLLMClient(responses={})
    # 模拟：第一次返回改写后的问题
    call_count = {"n": 0}

    def fake_complete(messages, temperature=None, max_tokens=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "高血压的治疗药物有哪些？"
        return "基于来源，治疗药物包括 X [S1]."

    llm.complete = fake_complete  # type: ignore[method-assign]
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(("什么是高血压？", "高血压是一种慢性病。"))
    msg = svc.ask("那治疗药物呢？", history=history)
    assert msg.metadata.get("standalone_query") == "高血压的治疗药物有哪些？"
    # 检索时被传入的是改写后的问题
    assert retriever.last_query == "高血压的治疗药物有哪些？"


# ---------------------------------------------------------------------------
# 29. 改写为空回退原问题
# ---------------------------------------------------------------------------
def test_rewrite_empty_fallback(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    # 第一次（改写）返回空 → 回退原 query；第二次（回答）返回正常答案
    call_count = {"n": 0}

    def fake_complete(messages, temperature=None, max_tokens=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ""
        return "答案 [S1]."

    llm = FakeLLMClient()
    llm.complete = fake_complete  # type: ignore[method-assign]
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(("什么是高血压？", "高血压是一种慢性病。"))
    msg = svc.ask("治疗药物？", history=history)
    # 改写为空 → 回退原问题
    assert msg.metadata.get("standalone_query") == "治疗药物？"
    assert "[S1]" in msg.content


def test_rewrite_looks_like_answer_fallback(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    """改写结果以「根据」开头 → 回退原问题。"""
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="根据以上信息，高血压是慢性病。")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(("什么是高血压？", "高血压是一种慢性病。"))
    msg = svc.ask("Q?", history=history)
    assert msg.metadata.get("standalone_query") == "Q?"


# ---------------------------------------------------------------------------
# 30. 空 query
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["", "   ", "\n\n\t  "])
def test_empty_query(
    settings: Settings, sample_retrieved: list[RetrievedChunk], bad: str
):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient()
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)
    with pytest.raises(QueryValidationError):
        svc.ask(bad)


# ---------------------------------------------------------------------------
# 31. 每轮重新检索
# ---------------------------------------------------------------------------
def test_each_turn_retrieves(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="A [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    svc.ask("Q1")
    svc.ask("Q2")
    svc.ask("Q3")
    assert retriever.call_count == 3


# ---------------------------------------------------------------------------
# 32 & 33. 空检索固定拒答
# ---------------------------------------------------------------------------
def test_empty_retrieval_fixed_refuse(
    settings: Settings
):
    retriever = FakeRetriever(results=[])
    llm = FakeLLMClient(default_response="SHOULD NOT BE CALLED")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    assert msg.content == NO_EVIDENCE_REPLY
    assert msg.citations == []
    assert msg.metadata.get("retrieval_count") == 0
    # 33: 不调用回答 LLM
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# 34. 正常答案和合法引用
# ---------------------------------------------------------------------------
def test_normal_answer_with_valid_citations(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="高血压是慢性病 [S1]，需监测 [S2].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    assert "[S1]" in msg.content
    assert "[S2]" in msg.content
    assert msg.citations == sample_retrieved


# ---------------------------------------------------------------------------
# 35. 非法引用被移除
# ---------------------------------------------------------------------------
def test_illegal_citations_removed(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    # S1 合法，S99 非法
    llm = FakeLLMClient(default_response="高血压 [S1] 引用 [S99] 不存在 [S2].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    assert "[S1]" in msg.content
    assert "[S2]" in msg.content
    assert "[S99]" not in msg.content
    # metadata 应记录非法引用
    assert "S99" in msg.metadata.get("illegal_citations", [])


# ---------------------------------------------------------------------------
# 36. citations 只来自 Retriever
# ---------------------------------------------------------------------------
def test_citations_only_from_retriever(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    # LLM 试图报新的来源 "fake_doc.txt" / 页码 5 / 段落 10 等
    llm = FakeLLMClient(
        default_response="答案 [S1] 来源 fake_doc.txt 第 5 页 段落 10 [S2]."
    )
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    # citations 完全由 retriever 决定
    assert len(msg.citations) == 2
    for c in msg.citations:
        # 包含原始 chunk 对象（来自 retriever）
        assert c.chunk in sample_retrieved[0].chunk.__class__.__mro__ or c in sample_retrieved
    # content 不应包含 fake_doc.txt / 第 5 页（这是用户提示的，真实 LLM 可能自报）
    # 注意：这里我们没有让 PromptBuilder 过滤文件名/页码；只验证 citations 不被 LLM 注入
    # content 中含 [S1]/[S2] 即可
    assert "[S1]" in msg.content
    assert "[S2]" in msg.content


# ---------------------------------------------------------------------------
# 37. 历史回答不作为回答来源
# ---------------------------------------------------------------------------
def test_history_answer_not_used_as_source(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    """验证回答阶段只引用 Retriever 返回的 [S#]，不引用历史。"""
    retriever = FakeRetriever(sample_retrieved)
    # LLM 回答不应包含历史中的内容
    llm = FakeLLMClient(default_response="只根据 [S1] 的内容回答.")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(
        ("什么是高血压？", "高血压的药物是历史回答里的旧信息。"),
        ("高血压有什么症状？", "头晕是症状。"),
    )
    msg = svc.ask("那用药呢？", history=history)
    # content 只来自 [S1]，不应含历史文本
    assert "历史回答" not in msg.content
    assert "旧信息" not in msg.content
    # citations 只包含答案中实际引用的 [S1]
    assert {c.citation_id for c in msg.citations} == {"S1"}


# ---------------------------------------------------------------------------
# 38. complete 异常映射
# ---------------------------------------------------------------------------
def test_complete_error_mapped(settings: Settings, sample_retrieved: list[RetrievedChunk]):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(
        default_response="",
        raise_on_complete=LLMError("LLM boom"),
    )
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    with pytest.raises(ChatGenerationError):
        svc.ask("Q?")


# ---------------------------------------------------------------------------
# 39. 流式 token 顺序
# ---------------------------------------------------------------------------
def test_stream_token_order(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["你", "好", " [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    tokens = [e.content for e in events if e.event_type == CHAT_EVENT_TOKEN]
    assert tokens == ["你", "好", " [S1]."]


# ---------------------------------------------------------------------------
# 40. 流式 sources 事件
# ---------------------------------------------------------------------------
def test_stream_sources_event(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["A [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    sources = [e for e in events if e.event_type == CHAT_EVENT_SOURCES]
    assert len(sources) == 1
    assert sources[0].citations == sample_retrieved
    assert sources[0].metadata.get("retrieval_count") == 2


# ---------------------------------------------------------------------------
# 41. 流式 done 事件
# ---------------------------------------------------------------------------
def test_stream_done_event_present(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["A [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE]
    assert len(done) == 1


# ---------------------------------------------------------------------------
# 42. done 事件包含最终 ChatMessage
# ---------------------------------------------------------------------------
def test_stream_done_contains_final_message(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["A [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    assert done.message is not None
    assert done.message.role == "assistant"
    assert done.message.content == "A [S1]."
    # done.citations 只含答案中实际使用的 [S1]
    assert {c.citation_id for c in done.message.citations} == {"S1"}
    assert "S1" in done.message.content


# ---------------------------------------------------------------------------
# 43. 流式非法引用最终清理
# ---------------------------------------------------------------------------
def test_stream_illegal_citations_cleaned(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)
    # 流式期间 LLM 仍会 emit 含 S99 的内容
    llm = StreamingFakeLLMClient(
        stream_chunks=["你好", " [S99] 再见 [S1]."]
    )
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    # done.message 的 content 中 [S99] 应被移除
    assert "[S99]" not in done.message.content
    # done.message.content 是最终答案
    assert done.message.content == "你好  再见 [S1]."  # [S99] 被替换为空 → 留有空格
    # metadata 应记录非法引用
    assert "S99" in done.message.metadata.get("illegal_citations", [])


# ---------------------------------------------------------------------------
# 44. 流式中途异常
# ---------------------------------------------------------------------------
def test_stream_midway_exception(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    retriever = FakeRetriever(sample_retrieved)

    def gen():
        yield "你"
        raise LLMError("LLM boom during stream")

    llm = FakeLLMClient()
    llm.stream = gen  # type: ignore[method-assign]
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    with pytest.raises(ChatGenerationError):
        list(svc.stream("Q?"))


# ---------------------------------------------------------------------------
# 45. reasoning 标签不会重新泄漏
# ---------------------------------------------------------------------------
def test_stream_reasoning_not_leaked(
    settings: Settings, sample_retrieved: list[RetrievedChunk]
):
    """LLM 流式输出如果仍带 <think> 标签，由 LLMClient 层负责过滤；
    ChatService 应原样透传已过滤内容。"""
    retriever = FakeRetriever(sample_retrieved)
    # 注意：FakeLLMClient.stream 直接 yield stream_chunks，不做过滤
    # 这里构造一个「不含标签」的输入，验证 ChatService 不引入标签
    llm = StreamingFakeLLMClient(stream_chunks=["纯文本", "回答 [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    # 全部 token 中不应含 think/analysis 标签
    tokens = [e.content for e in events if e.event_type == CHAT_EVENT_TOKEN]
    full = "".join(tokens)
    assert "<think>" not in full
    assert "</think>" not in full
    assert "<analysis>" not in full
    assert "</analysis>" not in full
    # done message 中也不应出现
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    assert "<think>" not in done.message.content


# ---------------------------------------------------------------------------
# 流式：检索为空时的事件顺序
# ---------------------------------------------------------------------------
def test_stream_empty_retrieval_event_order(settings: Settings):
    retriever = FakeRetriever(results=[])
    llm = StreamingFakeLLMClient(stream_chunks=[])  # 不会被调用
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    types = [e.event_type for e in events]
    # sources → token → done
    assert CHAT_EVENT_SOURCES in types
    assert CHAT_EVENT_TOKEN in types
    assert CHAT_EVENT_DONE in types
    # done.message 应该是固定拒答
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    assert done.message.content == NO_EVIDENCE_REPLY
    assert llm.call_count == 0


# ---------------------------------------------------------------------------
# 改写：LLM 错误时抛 QueryRewriteError
# ---------------------------------------------------------------------------
def test_rewrite_llm_error(settings: Settings, sample_retrieved: list[RetrievedChunk]):
    retriever = FakeRetriever(sample_retrieved)
    llm = FakeLLMClient(default_response="", raise_on_complete=LLMError("boom"))
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(("Q?", "A."))
    with pytest.raises(QueryRewriteError):
        svc.ask("follow-up", history=history)


# ---------------------------------------------------------------------------
# 检索时异常被包装为 ChatGenerationError
# ---------------------------------------------------------------------------
def test_retrieve_error_mapped(settings: Settings):
    retriever = FakeRetriever()
    retriever.raise_on_retrieve = RetrieverError("boom")
    llm = FakeLLMClient()
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    with pytest.raises(ChatGenerationError):
        svc.ask("Q?")


# ---------------------------------------------------------------------------
# 流式：rewrite 事件
# ---------------------------------------------------------------------------
def test_stream_rewrite_event(settings: Settings, sample_retrieved: list[RetrievedChunk]):
    retriever = FakeRetriever(sample_retrieved)
    call_count = {"n": 0}

    def fake_complete(messages, temperature=None, max_tokens=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "改写后的问题"
        return "Answer [S1]."

    llm = FakeLLMClient()
    llm.complete = fake_complete  # type: ignore[method-assign]
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    history = _make_history(("什么是 X？", "X 是 Y。"))
    events = list(svc.stream("那 Z 呢？", history=history))
    rewrites = [e for e in events if e.event_type == CHAT_EVENT_REWRITE]
    assert len(rewrites) == 1
    assert rewrites[0].content == "改写后的问题"


# ===========================================================================
# 7.2 引用一致性：最终 citations 只含答案中实际出现的 [S#]
# ===========================================================================
def _build_retrieved_3() -> list[RetrievedChunk]:
    return _build_retrieved([
        ("c1", "d1", "高血压是慢性病。", 0.95, "S1"),
        ("c2", "d1", "高血压需监测血压。", 0.88, "S2"),
        ("c3", "d1", "高血压可引发中风。", 0.80, "S3"),
    ])


# ---------------------------------------------------------------------------
# 1. 候选 S1/S2，答案只引用 [S1] → 最终 citations 只含 S1
# ---------------------------------------------------------------------------
def test_citations_filter_only_used(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="高血压是慢性病 [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    assert {c.citation_id for c in msg.citations} == {"S1"}
    assert msg.metadata.get("candidate_citation_ids") == ["S1", "S2", "S3"]
    assert msg.metadata.get("used_citation_ids") == ["S1"]


# ---------------------------------------------------------------------------
# 2. 答案只引用 [S2] → citations 只含 S2
# ---------------------------------------------------------------------------
def test_citations_filter_only_s2(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="高血压需监测 [S2].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    assert {c.citation_id for c in msg.citations} == {"S2"}


# ---------------------------------------------------------------------------
# 3. 答案引用顺序 [S2] 后 [S1] → citations 顺序为 S2、S1
# ---------------------------------------------------------------------------
def test_citations_preserve_answer_order(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="监测 [S2] 然后 [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    ids = [c.citation_id for c in msg.citations]
    assert ids == ["S2", "S1"]


# ---------------------------------------------------------------------------
# 4. 重复引用 [S1][S1] 只保留一次
# ---------------------------------------------------------------------------
def test_citations_dedupe(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="高血压 [S1] 又 [S1] 又 [S1].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    assert {c.citation_id for c in msg.citations} == {"S1"}
    assert len(msg.citations) == 1


# ---------------------------------------------------------------------------
# 5. 答案含非法 [S99]  → 过滤后不进入 citations
# ---------------------------------------------------------------------------
def test_illegal_citation_not_in_final_citations(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="高血压 [S1] 引用 [S99].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    # [S99] 被移除；最终 citations 只含 S1
    assert {c.citation_id for c in msg.citations} == {"S1"}
    assert "S99" in msg.metadata.get("illegal_citations", [])


# ---------------------------------------------------------------------------
# 6. 答案无任何引用 → citations 空 + citation_warning
# ---------------------------------------------------------------------------
def test_no_citation_in_answer_sets_warning(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="高血压是慢性病。")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("高血压?")
    assert msg.citations == []
    assert msg.metadata.get("citation_warning") == "answer_has_no_citation_reference"
    # candidate 仍保留
    assert msg.metadata.get("candidate_citation_ids") == ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# 7. 固定拒答语 → citations 始终为空
# ---------------------------------------------------------------------------
def test_fixed_refusal_citations_empty(settings: Settings):
    from rag.prompt_builder import NO_EVIDENCE_REPLY
    retriever = FakeRetriever(_build_retrieved_3())
    # LLM 试图说点东西，但 retriever 返回 0（触发固定拒答）
    retriever2 = FakeRetriever(results=[])
    llm = FakeLLMClient(default_response="不可能被调用")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever2, pb, llm, settings)

    msg = svc.ask("Q?")
    assert msg.content == NO_EVIDENCE_REPLY
    assert msg.citations == []
    # candidate 留空（因为是空检索）
    assert msg.metadata.get("candidate_citation_ids") == []


# ---------------------------------------------------------------------------
# 7b. 检索非空但 LLM 返回固定拒答语（极端场景）→ citations 仍空
# ---------------------------------------------------------------------------
def test_refusal_text_with_candidates_citations_empty(settings: Settings):
    from rag.prompt_builder import NO_EVIDENCE_REPLY
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    # LLM 仍按 prompt 返回拒答
    llm = FakeLLMClient(default_response=NO_EVIDENCE_REPLY)
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    assert msg.content == NO_EVIDENCE_REPLY
    assert msg.citations == []
    # candidate 仍记录
    assert msg.metadata.get("candidate_citation_ids") == ["S1", "S2", "S3"]


# ---------------------------------------------------------------------------
# 8. 空检索拒答 citations 空
# ---------------------------------------------------------------------------
def test_empty_retrieval_citations_empty(settings: Settings):
    from rag.prompt_builder import NO_EVIDENCE_REPLY
    retriever = FakeRetriever(results=[])
    llm = FakeLLMClient()
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    assert msg.content == NO_EVIDENCE_REPLY
    assert msg.citations == []


# ---------------------------------------------------------------------------
# 9. 流式 done 事件只含实际使用的引用
# ---------------------------------------------------------------------------
def test_stream_done_citations_filtered(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["高血压 [S2] 监测"])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    assert {c.citation_id for c in done.message.citations} == {"S2"}


# ---------------------------------------------------------------------------
# 10. 流式拒答 done.citations 为空
# ---------------------------------------------------------------------------
def test_stream_refusal_done_citations_empty(settings: Settings):
    from rag.prompt_builder import NO_EVIDENCE_REPLY
    retriever = FakeRetriever(results=[])
    llm = StreamingFakeLLMClient(stream_chunks=[])  # 不会被调用
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    assert done.message.content == NO_EVIDENCE_REPLY
    assert done.message.citations == []


# ---------------------------------------------------------------------------
# 11. sources 候选事件不影响最终 citations
# ---------------------------------------------------------------------------
def test_sources_event_has_candidates_done_has_filtered(settings: Settings):
    retrieved = _build_retrieved_3()
    retriever = FakeRetriever(retrieved)
    llm = StreamingFakeLLMClient(stream_chunks=["只引用 [S1]."])
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    events = list(svc.stream("Q?"))
    sources = [e for e in events if e.event_type == CHAT_EVENT_SOURCES][0]
    done = [e for e in events if e.event_type == CHAT_EVENT_DONE][0]
    # sources 带全部候选
    assert {c.citation_id for c in sources.citations} == {"S1", "S2", "S3"}
    # done 只含实际引用的
    assert {c.citation_id for c in done.message.citations} == {"S1"}


# ---------------------------------------------------------------------------
# 12. [S10] 不误匹配 [S1]
# ---------------------------------------------------------------------------
def test_s10_not_matched_as_s1(settings: Settings):
    # 构造含 S1 与 S10 的候选
    retrieved = _build_retrieved([
        ("c1", "d1", "短文", 0.95, "S1"),
        ("c10", "d1", "长文", 0.80, "S10"),
    ])
    retriever = FakeRetriever(retrieved)
    llm = FakeLLMClient(default_response="参见 [S10].")
    pb = PromptBuilder(settings)
    svc = ChatService(retriever, pb, llm, settings)

    msg = svc.ask("Q?")
    # 最终 citations 应只含 S10
    assert {c.citation_id for c in msg.citations} == {"S10"}


# ---------------------------------------------------------------------------
# 13. PromptBuilder.select_cited_chunks 直接测试
# ---------------------------------------------------------------------------
def test_select_cited_chunks_direct(settings: Settings):
    pb = PromptBuilder(settings)
    retrieved = _build_retrieved_3()
    # 只引 S2
    out = pb.select_cited_chunks("引用 [S2].", retrieved)
    assert [c.citation_id for c in out] == ["S2"]
    # 引用顺序 [S3] 后 [S1]
    out2 = pb.select_cited_chunks("[S3] and [S1]", retrieved)
    assert [c.citation_id for c in out2] == ["S3", "S1"]
    # 重复 [S1][S1]
    out3 = pb.select_cited_chunks("[S1] [S1]", retrieved)
    assert [c.citation_id for c in out3] == ["S1"]
    # [S10] 不会匹配成 S1
    out4 = pb.select_cited_chunks("[S10]", retrieved)
    assert out4 == []
    # 固定拒答语
    out5 = pb.select_cited_chunks(NO_EVIDENCE_REPLY, retrieved)
    assert out5 == []
    # 空字符串
    out6 = pb.select_cited_chunks("", retrieved)
    assert out6 == []
    # 不修改原 RetrievedChunk
    out7 = pb.select_cited_chunks("[S1]", retrieved)
    assert out7[0] is retrieved[0]
