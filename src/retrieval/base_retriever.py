from abc import ABC, abstractmethod
from llama_index.core.schema import TextNode
from src.rag_chunker import RagChunkTextNodeConverter, RagChunk
from .retrieval_data_models import RetrievalResults


class BaseRetriever(ABC):
    ALLOWED_EXPANSION_STRATEGIES = {
        "none", "neighbors", "first_section", "all_sections"
    }

    def __init__(
            self,
            insert_metadata_into_text: bool = True,
            expansion_strategy: str = "none",
            top_k: int = 5,
    ):
        if expansion_strategy not in self.ALLOWED_EXPANSION_STRATEGIES:
            raise ValueError(f"Expansion strategy {expansion_strategy} is not supported.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        self.text_node_converter = RagChunkTextNodeConverter()

        self.nodes = None

        self.insert_metadata_into_text = insert_metadata_into_text
        self.top_k = top_k
        self.expansion_strategy = expansion_strategy

    def _to_nodes(self, items: list[RagChunk] | list[TextNode]) -> list[TextNode]:
        if not items:
            return []

        first_item = items[0]
        if isinstance(first_item, TextNode):
            return items

        return self.text_node_converter.rag_chunks_to_nodes(items, add_metadata_to_text=self.insert_metadata_into_text)

    @abstractmethod
    def index(
            self,
            items: list[RagChunk] | list[TextNode],
    ) -> None:
        pass

    @abstractmethod
    def retrieve(
            self,
            query: str,
    ) -> RetrievalResults:
        pass


    def _get_context_nodes(self, result_nodes: list[TextNode]) -> list[TextNode]:
        if self.expansion_strategy == "none":
            return result_nodes
        elif self.expansion_strategy in {"first_section", "all_sections"}:
            relevant_nodes = self._get_section_nodes(result_nodes)
        elif self.expansion_strategy == "neighbors":
            relevant_nodes = self._get_neighbours_nodes(result_nodes)
        else:
            raise ValueError("Unknown expansion strategy: ", self.expansion_strategy)
        return relevant_nodes


    def _get_neighbours_nodes(
            self,
            result_nodes: list[TextNode],
    ) -> list[TextNode]:
        if self.nodes is None:
            raise AttributeError("Please call index() before retrieving")

        node_id_to_idx = {
            self._get_chunk_id(node): idx
            for idx, node in enumerate(self.nodes)
        }
        selected_indices = set()
        for node in result_nodes:

            idx = node_id_to_idx.get(self._get_chunk_id(node))
            if idx is None:
                continue

            for neighbour_idx in [idx - 1, idx, idx + 1]:
                if 0 <= neighbour_idx < len(self.nodes):
                    neighbour = self.nodes[neighbour_idx]

                    same_source = (
                            neighbour.metadata.get("source")
                            == node.metadata.get("source")
                    )

                    if same_source:
                        selected_indices.add(neighbour_idx)
        return [
            self.nodes[idx]
            for idx in sorted(selected_indices)

        ]


    def _get_section_nodes(
            self,
            result_nodes: list[TextNode],
    ) -> list[TextNode]:
        if self.nodes is None:
            raise AttributeError("Please call index() before retrieving")

        if not result_nodes:
            return []

        if self.expansion_strategy == "first_section":
            first_node = result_nodes[0]
            first_section = first_node.metadata.get("section")

            if first_section is None or first_section == "":
                return self._get_neighbours_nodes(result_nodes)

            relevant_section_keys = {
                (first_node.metadata.get("source"), first_section)
            }

        elif self.expansion_strategy == "all_sections":
            nodes_with_sections = [
                node
                for node in result_nodes
                if node.metadata.get("section") is not None
                   and node.metadata.get("section") != ""
            ]

            if not nodes_with_sections:
                return self._get_neighbours_nodes(result_nodes)

            relevant_section_keys = {
                (node.metadata.get("source"), node.metadata.get("section"))
                for node in nodes_with_sections
            }

        else:
            return result_nodes

        relevant_nodes = [
            node
            for node in self.nodes
            if (
                   node.metadata.get("source"),
                   node.metadata.get("section"),
               ) in relevant_section_keys
        ]

        return relevant_nodes


    def _get_chunk_id(self, node: TextNode) -> str:
        return node.metadata.get("chunk_id") or node.node_id