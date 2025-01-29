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
    # Get nodes and predicates from request
    request_data = request.get("nodes", [])
    predicates = request.get("predicates", [])

    # Create mappings for node properties and IDs
    node_properties = {}
    node_id_mapping = {}  # Maps request node_ids to their types
    for node in request_data:
        node_type = node.get("type", "")
        node_id = node.get("node_id", "")
        properties = node.get("properties", {})
        node_properties[node_type] = properties
        node_id_mapping[node_id] = node_type

    # Find matching nodes based on properties
    nodes_by_type = defaultdict(list)
    for node in result_graph.nodes:
        node_type = node.data.get("type", "unknown")
        if node_type in node_properties:
            # Check if node matches the requested properties
            matches = True
            for prop, value in node_properties[node_type].items():
                if node.data.get(prop) != value:
                    matches = False
                    break
            if matches:
                nodes_by_type[node_type].append(node)

    # Find and filter connected nodes based on predicates
    filtered_nodes_by_type = defaultdict(list)
    for predicate in predicates:
        source_type = node_id_mapping.get(predicate["source"])
        target_type = node_id_mapping.get(predicate["target"])
        relationship_type = predicate["type"].lower().replace(" ", "_")

        # First, add all source nodes that match the properties
        if source_type in nodes_by_type:
            filtered_nodes_by_type[source_type].extend(nodes_by_type[source_type])

        # Then find all valid target nodes connected to these source nodes
        valid_target_nodes = []
        for edge in result_graph.edges:
            if edge.data.get("label") == relationship_type:
                for source_node in nodes_by_type[source_type]:
                    if edge.data["source"] == f"{source_type} {source_node.data['id']}":
                        target_node_id = edge.data["target"]
                        # Find the actual target node
                        for target_node in nodes_by_type[target_type]:
                            if (
                                f"{target_type} {target_node.data['id']}"
                                == target_node_id
                            ):
                                # Check if node is not already in the list
                                if target_node not in valid_target_nodes:
                                    valid_target_nodes.append(target_node)

        # Update filtered nodes with only the valid target nodes
        if target_type in nodes_by_type:
            filtered_nodes_by_type[target_type] = valid_target_nodes

    # Create parent nodes and process nodes
    modified_nodes = []
    parent_ids = {}

    # Create parent nodes for types with multiple nodes
    for node_type, nodes in filtered_nodes_by_type.items():
        if len(nodes) > 1:
            parent_id = f"n{nanoid.generate()}"
            parent_ids[node_type] = parent_id
            modified_nodes.append(
                Node(
                    data={
                        "id": parent_id,
                        "type": "parent",
                        "name": f"{len(nodes)} {node_type} nodes",
                    }
                )
            )

    # Process all nodes
    for node_type, nodes in filtered_nodes_by_type.items():
        for node in nodes:
            node_data = {
                "id": f"{node_type} {node.data['id']}",
                "type": node_type,
                "name": f"{node_type} {node.data['id']}",
            }

            # Copy specific properties based on node type
            if node_type == "gene":
                relevant_properties = [
                    "gene_name",
                    "gene_type",
                    "start",
                    "end",
                    "chr",
                    "strand",
                    "description",
                ]
                for prop in relevant_properties:
                    if prop in node.data:
                        node_data[prop] = node.data[prop]
            else:
                # For other node types, copy all properties except id, type, name
                for key, value in node.data.items():
                    if key not in ["id", "type", "name", "synonyms"]:
                        node_data[key] = value

            # Add parent reference if this type has a parent node
            if node_type in parent_ids:
                node_data["parent"] = parent_ids[node_type]

            modified_nodes.append(Node(data=node_data))

    # Create edges based on predicates
    new_edges = []
    for predicate in predicates:
        source_type = node_id_mapping.get(predicate["source"])
        target_type = node_id_mapping.get(predicate["target"])

        for source_node in filtered_nodes_by_type[source_type]:
            target = parent_ids.get(target_type)
            if target:
                edge_data = {
                    "edge_id": f"{source_type}_{predicate['type']}_{target_type}",
                    "label": predicate["type"].lower().replace(" ", "_"),
                    "source": f"{source_type} {source_node.data['id']}",
                    "target": target,
                    "id": f"e{nanoid.generate()}",
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

