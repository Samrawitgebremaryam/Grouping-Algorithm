from neo4j import GraphDatabase
from typing import Dict, Any


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()

    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data from Neo4j database"""
        with self.driver.session() as session:
            query = """
            MATCH (g:gene)-[r:transcribed_to]->(t:transcript)
            WITH collect(distinct g) as genes, collect(distinct t) as transcripts, 
                 collect(distinct {rel: r, start: g, end: t}) as rels
            RETURN genes as nodes, transcripts as more_nodes, rels as relationships
            """

            result = session.run(query)
            record = result.single()

            if not record:
                return {"nodes": [], "edges": []}

            nodes = []
            edges = []

            # Process all gene nodes
            for node in record["nodes"]:
                try:
                    node_labels = list(node.labels)
                    node_type = node_labels[0].lower()
                    node_properties = dict(node.items())
                    node_id = node_properties.get("id", "")

                    node_data = {
                        "id": f"{node_type} {node_id}",
                        "type": node_type,
                        "name": f"{node_type} {node_id}",
                    }
                    node_data.update(node_properties)
                    nodes.append({"data": node_data})
                except Exception as e:
                    print(f"Error processing gene node: {str(e)}")
                    continue

            # Process all transcript nodes
            for node in record["more_nodes"]:
                try:
                    node_labels = list(node.labels)
                    node_type = node_labels[0].lower()
                    node_properties = dict(node.items())
                    node_id = node_properties.get("id", "")

                    node_data = {
                        "id": f"{node_type} {node_id}",
                        "type": node_type,
                        "name": f"{node_type} {node_id}",
                    }
                    node_data.update(node_properties)
                    nodes.append({"data": node_data})
                except Exception as e:
                    print(f"Error processing transcript node: {str(e)}")
                    continue

            # Process relationships
            for rel_data in record["relationships"]:
                try:
                    rel = rel_data["rel"]
                    start_node = rel_data["start"]
                    end_node = rel_data["end"]

                    start_type = list(start_node.labels)[0].lower()
                    end_type = list(end_node.labels)[0].lower()

                    edges.append(
                        {
                            "data": {
                                "edge_id": f"{rel.type.lower()}_{start_node['id']}_{end_node['id']}",
                                "label": rel.type.lower(),
                                "source": f"{start_type} {start_node['id']}",
                                "target": f"{end_type} {end_node['id']}",
                            }
                        }
                    )
                except Exception as e:
                    print(f"Error processing relationship: {str(e)}")
                    continue

            return {"nodes": nodes, "edges": edges}
