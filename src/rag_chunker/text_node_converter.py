from llama_index.core.schema import TextNode

from .data_models import RagChunk


class RagChunkTextNodeConverter:
    """
    Converts project RagChunk objects to LlamaIndex TextNode objects.
    """

    def rag_chunk_to_node(
            self,
            chunk: RagChunk,
            add_metadata_to_text: bool = False,
    ) -> TextNode:
        return TextNode(
            id_=chunk.chunk_id,
            text=self._build_node_text(
                chunk=chunk,
                add_metadata_to_text=add_metadata_to_text,
            ),
            metadata=self._build_metadata(chunk),
        )

    def rag_chunks_to_nodes(
            self,
            chunks: list[RagChunk],
            add_metadata_to_text: bool = False,
    ) -> list[TextNode]:
        return [
            self.rag_chunk_to_node(
                chunk=chunk,
                add_metadata_to_text=add_metadata_to_text,
            )
            for chunk in chunks
        ]

    def _build_metadata(self, chunk: RagChunk) -> dict:
        return {
            "source": chunk.source,
            "chunk_id": chunk.chunk_id,
            "section": chunk.section,
            "pages": chunk.pages,
            "content_types": [
                content_type.name
                for content_type in chunk.content_types
            ],
            "asset_paths": [
                {
                    "path": asset.path,
                    "content": asset.content,
                    "type": asset.type.name,
                }
                for asset in chunk.asset_paths
            ],
        }

    def _build_node_text(
            self,
            chunk: RagChunk,
            add_metadata_to_text: bool = False,
    ) -> str:
        if not add_metadata_to_text:
            return chunk.content or ""

        content_type_names = [t.name for t in chunk.content_types]

        return (
            f"Документ: {chunk.source}\n"
            f"Раздел: {chunk.section}\n"
            f"Страницы: {chunk.pages}\n"
            f"Типы содержимого: {content_type_names}\n\n"
            f"{chunk.content or ''}"
        )
