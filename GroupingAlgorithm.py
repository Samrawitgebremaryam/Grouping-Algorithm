from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
from neo4j import GraphDatabase
import nanoid
import os

# Constants
MINIMUM_EDGES_TO_COLLAPSE = 2

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
        return cls(
            nodes=[Node(data=n["data"]) for n in data.get("nodes", [])],
            edges=[Edge(data=e["data"]) for e in data.get("edges", [])]
        )

    def to_dict(self) -> Dict:
        return {
            "nodes": [{"data": n.data} for n in self.nodes],
            "edges": [{"data": e.data} for e in self.edges]
        }

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str, max_retries: int = 3):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.max_retries = max_retries

    def close(self):
        if self.driver:
            self.driver.close()

    def get_graph_data(self, request_data: Dict, limit: int = 1000) -> Dict[str, Any]:
        try:
            with self.driver.session() as session:
                # Get available labels (node types)
                label_query = """
                CALL db.labels() YIELD label 
                RETURN collect(toLower(label)) as labels
                """
                labels_result = session.run(label_query).single()
                available_labels = labels_result["labels"] if labels_result else []
                print(f"Available labels: {available_labels}")

                # Get available relationship types
                rel_query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN collect(toLower(relationshipType)) as types
                """
                rel_result = session.run(rel_query).single()
                available_rels = rel_result["types"] if rel_result else []
                print(f"Available relationships: {available_rels}")

                # Build dynamic query based on node types and relationships in request data
                query_parts = []
                parameters = {}

                for node in request_data.get("nodes", []):
                    node_type = node.get("type", "").lower()
                    if node_type not in available_labels:
                        raise ValueError(f"Node type '{node_type}' is not available in the database")

                    properties = node.get("properties", {})
                    node_conditions = []
                    for prop, value in properties.items():
                        param_key = f"{node_type}_{prop}"
                        parameters[param_key] = value
                        node_conditions.append(f"toLower(n.{prop}) = toLower(${param_key})")
                    
                    if node_conditions:
                        query_parts.append(f"(n:{node_type}) WHERE " + " AND ".join(node_conditions))
                    else:
                        query_parts.append(f"(n:{node_type})")

                # Handle relationships in request data
                rel_query_parts = []
                for predicate in request_data.get("predicates", []):
                    rel_type = predicate.get("type", "").lower()
                    if rel_type not in available_rels:
                        raise ValueError(f"Relationship type '{rel_type}' is not available in the database")

                    source = predicate.get("source")
                    target = predicate.get("target")
                    rel_query_parts.append(f"({source})-[r:{rel_type}]->({target})")

                # Construct the final query
                query = "MATCH " + ", ".join(query_parts + rel_query_parts)
                query += " RETURN collect(distinct n) as nodes, collect(distinct r) as relationships LIMIT $limit"

                # Run the query
                result = session.run(query, **parameters, limit=limit)
                record = result.single()

                if not record:
                    return {"nodes": [], "edges": []}

                nodes = []
                edges = []

                # Process nodes
                for node in record["nodes"]:
                    node_properties = dict(node.items())
                    node_id = node_properties.get("id", str(node.id))
                    node_type = node_properties.get("label", "unknown")

                    nodes.append({
                        "data": {
                            "id": node_id,
                            "type": node_type,
                            **node_properties  # Include all node properties dynamically
                        }
                    })

                # Process relationships
                for rel in record["relationships"]:
                    start_id = rel.start_node.id
                    end_id = rel.end_node.id

                    edges.append({
                        "data": {
                            "id": f"e{nanoid.generate(size=10)}",
                            "label": rel.type,
                            "source": start_id,
                            "target": end_id,
                        }
                    })

                return {"nodes": nodes, "edges": edges}

        except Exception as e:
            print(f"Error getting graph data: {str(e)}")
            raise

def group_edges(result_graph: Graph, request: Dict) -> List[Dict]:
    """Group edges by edge_id and handle any node types"""
    edge_groups = defaultdict(list)

    for edge in result_graph.edges:
        edge_id = edge.data.get("edge_id")
        if edge_id:
            edge_groups[edge_id].append(edge)

    edge_groupings = []
    for edge_id, edges in edge_groups.items():
        if len(edges) >= MINIMUM_EDGES_TO_COLLAPSE:
            source_type, relationship, target_type = edge_id.split("_", 2)

            source_groups = {}
            target_groups = {}

            for edge in edges:
                source = edge.data.get("source")
                target = edge.data.get("target")

                if source not in source_groups:
                    source_groups[source] = []
                if target not in target_groups:
                    target_groups[target] = []

                source_groups[source].append(edge)
                target_groups[target].append(edge)

            grouped_by = "target" if len(source_groups) > len(target_groups) else "source"
            groups = target_groups if grouped_by == "target" else source_groups

            edge_groupings.append({
                "count": len(edges),
                "edge_id": edge_id,
                "edge_type": edges[0].data.get("label"),
                "grouped_by": grouped_by,
                "groups": groups,
            })

    return edge_groupings

def group_graph(result_graph: Graph, request: Dict) -> Graph:
    """Group nodes and edges based on request data"""
    new_graph = Graph(nodes=[], edges=[])
    
    # Group nodes and edges as per the request
    for node in result_graph.nodes:
        new_graph.nodes.append(node)

    for edge in result_graph.edges:
        new_graph.edges.append(edge)

    return new_graph
