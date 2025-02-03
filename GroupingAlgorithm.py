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
                # Get available labels
                label_query = """
                CALL db.labels() YIELD label 
                RETURN collect(toLower(label)) as labels
                """
                labels_result = session.run(label_query).single()
                available_labels = labels_result["labels"] if labels_result else []
                print(f"Available labels: {available_labels}")

                # Get relationship types
                rel_query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN collect(toLower(relationshipType)) as types
                """
                rel_result = session.run(rel_query).single()
                available_rels = rel_result["types"] if rel_result else []

                # Get properties from request for each node type
                node_properties = {}
                for node in request_data.get("nodes", []):
                    node_type = node.get("type", "").lower()
                    properties = node.get("properties", {})
                    if node_type:
                        node_properties[node_type] = properties

                # Get gene name from properties
                gene_properties = {}
                for node in request_data.get("nodes", []):
                    if node.get("type", "").lower() == "gene":
                        gene_properties = node.get("properties", {})
                        break
                
                gene_name = gene_properties.get("gene_name")
                
                if not gene_name:
                    raise ValueError("gene_name is required in the request")

                query = """
                MATCH (g:gene)
                WHERE toLower(g.gene_name) = toLower($gene_name)
                WITH g
                MATCH (g)-[r]->(t:transcript)
                RETURN 
                    collect(distinct g) as nodes,
                    collect(distinct t) as transcripts,
                    collect(distinct {rel: r, start: g, end: t}) as relationships
                LIMIT $limit
                """

                result = session.run(query, gene_name=gene_name, limit=limit)
                record = result.single()

                if not record:
                    return {"nodes": [], "edges": []}

                nodes = []
                edges = []

                # Process gene nodes
                for node in record["nodes"]:
                    node_properties = dict(node.items())
                    node_id = node_properties.get("id", str(node.id))
                    nodes.append({
                        "data": {
                            "id": f"gene {node_id}",
                            "type": "gene",
                            "gene_name": node_properties.get("gene_name", ""),
                            "gene_type": node_properties.get("gene_type", ""),
                            "start": str(node_properties.get("start")),
                            "end": str(node_properties.get("end")),
                            "label": "gene",
                            "chr": node_properties.get("chr"),
                            "name": f"gene {node_id}",
                        }
                    })

                # Process transcript nodes
                for node in record["transcripts"]:
                    node_properties = dict(node.items())
                    node_id = node_properties.get("id", str(node.id))
                    nodes.append({
                        "data": {
                            "id": f"transcript {node_id}",
                            "type": "transcript",
                            "gene_name": gene_name,
                            "transcript_id": node_properties.get("transcript_id", ""),
                            "transcript_name": node_properties.get("transcript_name", ""),
                            "start": str(node_properties.get("start")),
                            "end": str(node_properties.get("end")),
                            "label": "transcript",
                            "transcript_type": node_properties.get("transcript_type", ""),
                            "chr": node_properties.get("chr"),
                            "name": f"transcript {node_id}",
                            "parent": ""  # This will be filled in by group_graph
                        }
                    })

                # Process relationships
                for rel_data in record["relationships"]:
                    rel = rel_data["rel"]
                    start_node = rel_data["start"]
                    end_node = rel_data["end"]

                    start_id = start_node.get("id", str(start_node.id))
                    end_id = end_node.get("id", str(end_node.id))

                    edges.append({
                        "data": {
                            "edge_id": f"gene_transcribed_to_transcript",
                            "label": "transcribed_to",
                            "source": f"gene {start_id}",
                            "target": f"transcript {end_id}",
                            "id": f"e{nanoid.generate(size=10)}",
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
    
    # Get all transcript nodes
    transcript_nodes = [node for node in result_graph.nodes if node.data["type"] == "transcript"]
    gene_nodes = [node for node in result_graph.nodes if node.data["type"] == "gene"]
    
    if len(transcript_nodes) >= MINIMUM_EDGES_TO_COLLAPSE:
        # Create parent node with nanoid
        parent_id = nanoid.generate(size=10)
        parent_node = Node(data={
            "id": parent_id,
            "type": "parent",
            "name": f"{len(transcript_nodes)} transcript nodes"
        })
        
        # Add parent node first
        new_graph.nodes.append(parent_node)
        
        # Add gene nodes
        new_graph.nodes.extend(gene_nodes)
        
        # Add transcript nodes with parent reference
        for node in transcript_nodes:
            node.data["parent"] = parent_id
            new_graph.nodes.append(node)
        
        # Create single edge from gene to parent
        if gene_nodes:
            new_edge = Edge(data={
                "id": f"e{nanoid.generate(size=10)}",
                "edge_id": "gene_transcribed_to_transcript",
                "label": "transcribed_to",
                "source": gene_nodes[0].data["id"],
                "target": parent_id
            })
            new_graph.edges = [new_edge]
    else:
        return result_graph
    
    return new_graph
