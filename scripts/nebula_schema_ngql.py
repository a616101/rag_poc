"""
輸出 NebulaGraph GraphRAG schema 的 nGQL（不直接連線執行）。

用法：
  python -m scripts.nebula_schema_ngql > nebula_schema.ngql
"""

from chatbot_rag.services.nebula_graph_service import nebula_graph_service


def main() -> None:
    for stmt in nebula_graph_service.ensure_schema_ngql():
        print(stmt)


if __name__ == "__main__":
    main()




