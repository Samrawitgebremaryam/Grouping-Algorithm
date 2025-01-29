from typing import Dict, List, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import nanoid
import json
from datetime import datetime
from flask import Flask, request, jsonify


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
        nodes = [Node(data=n["data"]) for n in data.get("nodes", [])]
        edges = [Edge(data=e["data"]) for e in data.get("edges", [])]
        return cls(nodes=nodes, edges=edges)

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
        type_counts[node_type] += 1
    return type_counts


def group_graph(result_graph: Graph, request: Dict) -> Graph:
    # Get nodes of requested types from the request
    request_data = request.get("nodes", [])
    requested_types = {node.get("type", "") for node in request_data}
    
    # Create a mapping of properties from request
    node_properties = {}
    for node in request_data:
        node_type = node.get("type", "")
        properties = node.get("properties", {})
        node_properties[node_type] = properties

    # First find the SNAP25 gene nodes
    gene_nodes = []
    for node in result_graph.nodes:
        if (node.data.get("type") == "gene" and 
            node.data.get("gene_name") == node_properties["gene"]["gene_name"]):
            gene_nodes.append(node)

    # Find transcripts connected to SNAP25 genes
    connected_transcripts = set()
    for edge in result_graph.edges:
        if (edge.data.get("label") == "transcribed_to" and 
            any(f"gene {gene.data['id']}" == edge.data["source"] for gene in gene_nodes)):
            connected_transcripts.add(edge.data["target"])

    # Group nodes by type and filter by properties and connections
    nodes_by_type = defaultdict(list)
    for node in result_graph.nodes:
        node_type = node.data.get("type", "unknown")
        if node_type == "gene" and node in gene_nodes:
            nodes_by_type["gene"].append(node)
        elif node_type == "transcript" and f"transcript {node.data['id']}" in connected_transcripts:
            nodes_by_type["transcript"].append(node)

    # Create parent node for transcripts
    parent_id = f"n{nanoid.generate()}"
    modified_nodes = []
    
    # Add parent node for transcripts with correct count
    if nodes_by_type["transcript"]:
        modified_nodes.append(Node(data={
            "id": parent_id,
            "type": "parent",
            "name": f"{len(nodes_by_type['transcript'])} transcript nodes"
        }))

    # Process gene nodes
    for node in nodes_by_type["gene"]:
        node_data = {
            "id": f"gene {node.data['id']}",
            "type": "gene",
            "gene_name": node_properties["gene"]["gene_name"],
            "gene_type": node.data.get("gene_type", ""),
            "start": node.data.get("start", ""),
            "end": node.data.get("end", ""),
            "label": "gene",
            "chr": node.data.get("chr", ""),
            "name": f"gene {node.data['id']}"
        }
        modified_nodes.append(Node(data=node_data))

    # Process transcript nodes
    for node in nodes_by_type["transcript"]:
        node_data = {
            "id": f"transcript {node.data['id']}",
            "type": "transcript",
            "gene_name": node_properties["gene"]["gene_name"],
            "transcript_id": node.data.get("transcript_id", ""),
            "transcript_name": node.data.get("transcript_name", ""),
            "start": node.data.get("start", ""),
            "end": node.data.get("end", ""),
            "label": "transcript",
            "transcript_type": node.data.get("transcript_type", ""),
            "chr": node.data.get("chr", ""),
            "name": f"transcript {node.data['id']}",
            "parent": parent_id
        }
        modified_nodes.append(Node(data=node_data))

    # Create edges
    new_edges = []
    for gene_node in nodes_by_type["gene"]:
        edge_data = {
            "edge_id": "gene_transcribed_to_transcript",
            "label": "transcribed_to",
            "source": f"gene {gene_node.data['id']}",
            "target": parent_id,
            "id": f"e{nanoid.generate()}"
        }
        new_edges.append(Edge(data=edge_data))

    return Graph(nodes=modified_nodes, edges=new_edges)


if __name__ == "__main__":
    # Create graph instance
    input_graph = Graph.from_dict(oldGraph)

    # Process the graph
    grouped_graph = group_graph(input_graph, request)

    # Print results
    print(json.dumps(grouped_graph.to_dict(), indent=2))
