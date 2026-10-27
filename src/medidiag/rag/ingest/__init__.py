"""多格式文档解析、清洗与去重（M5）。

所有解析器输出同一个中间结构：文档（Document）→ 章节（Section）→ 元素（Element），
元素类型为标题、段落、列表、表格。后续分块、元数据与索引只读这个结构，不关心原始格式。
设计参考 Unstructured 的元素划分。
"""

from medidiag.rag.ingest.document import Document, DocumentMetadata, Element, Section

__all__ = ["Document", "DocumentMetadata", "Element", "Section"]
