from typing import Dict, List, Set, Any, Tuple, TypedDict, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import nanoid
import json
from datetime import datetime
from flask import Flask, request, jsonify
from functools import lru_cache


@dataclass
class Edge:
    data: Dict[str, Any]


@dataclass
class Node:
    data: Dict[str, Any]


@dataclass
class Graph:
    nodes: List[Node]
    edges: List[Edge]

    @classmethod
    def from_dict(cls, data: Dict) -> "Graph":
        def process_chunks(items, processor, chunk_size=1000):
            return [
                processor(item)
                for chunk in (
                    items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
                )
                for item in chunk
            ]

        return cls(
            nodes=process_chunks(data.get("nodes", []), lambda n: Node(data=n["data"])),
            edges=process_chunks(data.get("edges", []), lambda e: Edge(data=e["data"])),
        )

    def to_dict(self) -> Dict:
        return {
            "nodes": [{"data": n.data} for n in self.nodes],
            "edges": [{"data": e.data} for e in self.edges],
        }


def count_nodes_by_type(nodes):
    """Helper function to count nodes by their type"""
    type_counts = defaultdict(int)
    for node in nodes:
        node_type = node.data.get("type", "unknown")
        type_counts[node_type] = 1
    return type_counts


def count_nodes_and_edges(filtered_nodes_by_type: Dict, edges_by_type: Dict) -> Dict:
    """Helper function to count nodes and edges with their labels"""
    node_type_counts = Counter(
        node.data.get("type", "unknown")
        for nodes in filtered_nodes_by_type.values()
        for node in nodes
    )

    edge_label_counts = Counter(
        edge.data.get("label")
        for edges in edges_by_type.values()
        for edge in edges
        if edge.data.get("label")
    )

    return {
        "node_count": sum(node_type_counts.values()),
        "edge_count": sum(edge_label_counts.values()),
        "node_count_by_label": [
            {"label": label, "count": count}
            for label, count in node_type_counts.items()
        ],
        "edge_count_by_label": [
            {"label": label, "count": count}
            for label, count in edge_label_counts.items()
        ],
    }


class NodeData(TypedDict):
    id: str
    type: str
    name: str
    label: str
    parent: Optional[str]


def process_node_properties(node: Node, node_type: str) -> NodeData:
    # Add type validation
    if not isinstance(node, Node) or not isinstance(node_type, str):
        raise ValueError("Invalid input types")
    # Cache node.data to avoid multiple dictionary lookups
    node_data_cache = node.data
    node_data = {
        "id": node_data_cache["id"],
        "type": node_type,
        "name": node_data_cache["id"],
    }

    # Define property mappings for each node type
    property_mappings = {
        "gene": [
            "gene_name",
            "gene_type",
            "start",
            "end",
            "chr",
            "strand",
            "description",
        ],
        "transcript": [
            "transcript_id",
            "transcript_type",
            "transcript_name",
            "start",
            "end",
            "chr",
            "gene_name",
        ],
        "promoter": ["start", "end", "chr"],
        "enhancer": ["start", "end", "chr", "data_source", "enhancer_id"],
        "pathway": ["pathway_name"],
        "protein": ["protein_name", "uniprot_id"],
        "snp": ["position", "chr", "ref_allele", "alt_allele"],
        "super_enhancer": ["start", "end", "chr", "data_source", "super_enhancer_id"],
        "exon": ["start", "end", "chr", "rank", "phase"],
    }

    # Copy relevant properties
    relevant_properties = property_mappings.get(node_type, [])
    for prop in relevant_properties:
        if prop in node_data_cache:
            node_data[prop] = node_data_cache[prop]

    node_data["label"] = node_type
    return node_data


def get_gene_name_for_transcript(
    node_data: Dict, node: Node, result_graph: Graph
) -> Dict:
    """Get gene name for transcript nodes from connected gene nodes"""
    if "gene_name" not in node_data:
        for edge in result_graph.edges:
            if edge.data["target"] == f"transcript {node.data['id']}" and edge.data[
                "source"
            ].startswith("gene "):
                gene_id = edge.data["source"].replace("gene ", "")
                for gene_node in result_graph.nodes:
                    if (
                        gene_node.data["type"] == "gene"
                        and gene_node.data["id"] == gene_id
                        and "gene_name" in gene_node.data
                    ):
                        node_data["gene_name"] = gene_node.data["gene_name"]
                        break
                break
    return node_data


def create_node_mappings(request_data: List[Dict]) -> Tuple[Dict, Dict]:
    """Create mappings for node properties and IDs"""
    node_properties = {}
    node_id_mapping = {}
    for node in request_data:
        node_type = node.get("type", "").lower()
        node_id = node.get("node_id", "")
        properties = node.get("properties", {})
        node_properties[node_type] = properties
        node_id_mapping[node_id] = node_type
    return node_properties, node_id_mapping


def find_matching_nodes(
    result_graph: Graph, node_properties: Dict
) -> Dict[str, List[Node]]:
    """Find nodes matching the requested properties"""
    nodes_by_type = defaultdict(list)
    for node in result_graph.nodes:
        node_type = node.data.get("type", "unknown").lower()
        if node_type in node_properties:
            if matches_properties(node, node_properties[node_type]):
                nodes_by_type[node_type].append(node)
    return nodes_by_type


def matches_properties(node: Node, properties: Dict) -> bool:
    # Cache node data and use any() for early exit
    node_data = node.data
    return not any(
        (
            isinstance(value, str)
            and isinstance(node_data.get(prop), str)
            and value.lower() != node_data[prop].lower()
        )
        or (node_data.get(prop) != value)
        for prop, value in properties.items()
    )


def create_parent_nodes(node_groups: Dict) -> Tuple[List[Node], Dict]:
    """Create parent nodes for groups"""
    modified_nodes = []
    parent_ids = {}

    for (node_type, pattern), nodes in node_groups.items():
        if len(nodes) > 1:
            parent_id = f"n{nanoid.generate(size=10)}"
            parent_ids[(node_type, pattern)] = parent_id
            modified_nodes.append(
                Node(
                    data={
                        "id": parent_id,
                        "type": "parent",
                        "name": f"{len(nodes)} {node_type} nodes",
                    }
                )
            )
    return modified_nodes, parent_ids


def create_edges(
    nodes_by_type: Dict, predicates: List[Dict], node_id_mapping: Dict, parent_ids: Dict
) -> List[Edge]:
    """Create edges based on predicates"""
    new_edges = []
    seen_edges = set()

    for predicate in predicates:
        source_type = node_id_mapping.get(predicate["source"])
        target_type = node_id_mapping.get(predicate["target"])
        relationship_type = predicate["type"].lower().replace(" ", "_")
        edge_id = f"{source_type}_{relationship_type}_{target_type}"

        new_edges.extend(
            create_edges_for_source_nodes(
                nodes_by_type[source_type],
                nodes_by_type[target_type],
                source_type,
                target_type,
                relationship_type,
                parent_ids,
                seen_edges,
            )
        )

    return new_edges


def create_edges_for_source_nodes(
    source_nodes: List[Node],
    target_nodes: List[Node],
    source_type: str,
    target_type: str,
    relationship_type: str,
    parent_ids: Dict,
    seen_edges: Set,
) -> List[Edge]:
    """Create edges between source nodes and target nodes/parent nodes"""
    # Pre-compute parent mapping for better performance
    parent_mapping = {t: pid for (t, _), pid in parent_ids.items()}
    parent_id = parent_mapping.get(target_type)

    # Batch edge creation
    edges = []
    edge_data_list = [
        {
            "edge_id": f"{source_type}_{relationship_type}_{target_type}",
            "label": relationship_type,
            "source": source_node.data["id"],
            "target": parent_id,
            "id": f"e{nanoid.generate(size=10)}",
        }
        for source_node in source_nodes
        if parent_id
        and (source_node.data["id"], parent_id, relationship_type) not in seen_edges
    ]

    edges.extend([Edge(data=data) for data in edge_data_list])
    seen_edges.update(
        (data["source"], data["target"], relationship_type) for data in edge_data_list
    )
    return edges


def create_node_groups(
    nodes_by_type: Dict[str, List[Node]], *args
) -> Dict[Tuple[str, str], List[Node]]:
    # Use dict comprehension for better performance
    return {
        (node_type, f"{node_type}_default"): nodes
        for node_type, nodes in nodes_by_type.items()
        if nodes  # Skip empty lists
    }


def group_graph(result_graph: Graph, request: Dict) -> Graph:
    """Main function to group nodes and create the graph"""
    request_data = request.get("nodes", [])
    predicates = request.get("predicates", [])

    # Create mappings and find matching nodes
    node_properties, node_id_mapping = create_node_mappings(request_data)
    nodes_by_type = find_matching_nodes(result_graph, node_properties)

    # Create node groups and patterns
    node_groups = create_node_groups(nodes_by_type)

    # Create parent nodes and process all nodes
    modified_nodes, parent_ids = create_parent_nodes(node_groups)

    # Process individual nodes
    for node_type, nodes in nodes_by_type.items():
        for node in nodes:
            node_data = process_node_properties(node, node_type)
            if node_type == "transcript":
                node_data = get_gene_name_for_transcript(node_data, node, result_graph)

            # Add parent reference
            for (t, pattern), parent_id in parent_ids.items():
                if t == node_type:
                    node_data["parent"] = parent_id
                    break

            modified_nodes.append(Node(data=node_data))

    # Create edges
    new_edges = create_edges(nodes_by_type, predicates, node_id_mapping, parent_ids)

    # Create final graph
    final_graph = Graph(nodes=modified_nodes, edges=new_edges)

    # Add counts
    graph_dict = final_graph.to_dict()
    graph_dict.update(count_nodes_and_edges(nodes_by_type, defaultdict(list)))

    return Graph.from_dict(graph_dict)


@lru_cache(maxsize=1000)
def get_edge_id(source_type: str, relationship_type: str, target_type: str) -> str:
    return f"{source_type}_{relationship_type}_{target_type}"
