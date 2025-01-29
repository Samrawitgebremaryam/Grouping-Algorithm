from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from typing import Dict, Any
import time


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str, max_retries: int = 3):
        self.uri = uri
        self.user = user
        self.password = password
        self.max_retries = max_retries
        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection with retry mechanism"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                self.driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password),
                    max_connection_lifetime=30,  # 30 seconds
                )
                # Verify connection
                self.driver.verify_connectivity()
                return
            except (ServiceUnavailable, SessionExpired) as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise Exception(
                        f"Failed to connect to Neo4j after {self.max_retries} attempts: {str(e)}"
                    )
                time.sleep(1)  # Wait 1 second before retrying

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()

    def get_graph_data(self) -> Dict[str, Any]:
        """Get graph data from Neo4j database with retry mechanism"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                with self.driver.session() as session:
                    print("Executing Neo4j query...")
                    query = """
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m)
                    WITH collect(distinct n) as nodes,
                         collect(distinct {rel: r, start: n, end: m}) as relationships
                    RETURN nodes, relationships
                    """
                    print(f"Query: {query}")

                    result = session.run(query)
                    print("\nQuery executed successfully")

                    record = result.single()
                    if not record:
                        print("No data found in database")
                        return {"nodes": [], "edges": []}

                    print(
                        f"\nFound {len(record['nodes'])} nodes and {len(record['relationships'])} relationships"
                    )

                    nodes = []
                    edges = []

                    # Process all nodes
                    for node in record["nodes"]:
                        try:
                            node_labels = list(node.labels)
                            node_type = (
                                node_labels[0].lower() if node_labels else "unknown"
                            )
                            node_properties = dict(node.items())
                            node_id = node_properties.get("id", str(node.id))

                            node_data = {
                                "id": f"{node_type} {node_id}",
                                "type": node_type,
                                "name": node_properties.get(
                                    "name", f"{node_type} {node_id}"
                                ),
                            }
                            node_data.update(node_properties)
                            nodes.append({"data": node_data})
                        except Exception as e:
                            print(f"Error processing node: {str(e)}")
                            continue

                    # Process all relationships
                    for rel_data in record["relationships"]:
                        try:
                            rel = rel_data["rel"]
                            if rel is None:
                                continue

                            start_node = rel_data["start"]
                            end_node = rel_data["end"]

                            if start_node is None or end_node is None:
                                continue

                            start_type = (
                                list(start_node.labels)[0].lower()
                                if list(start_node.labels)
                                else "unknown"
                            )
                            end_type = (
                                list(end_node.labels)[0].lower()
                                if list(end_node.labels)
                                else "unknown"
                            )

                            start_id = start_node.get("id", str(start_node.id))
                            end_id = end_node.get("id", str(end_node.id))

                            edges.append(
                                {
                                    "data": {
                                        "edge_id": f"{rel.type.lower()}_{start_id}_{end_id}",
                                        "label": rel.type.lower(),
                                        "source": f"{start_type} {start_id}",
                                        "target": f"{end_type} {end_id}",
                                        "id": f"e{str(rel.id)}",
                                    }
                                }
                            )
                        except Exception as e:
                            print(f"Error processing relationship: {str(e)}")
                            continue

                    return {"nodes": nodes, "edges": edges}

            except (ServiceUnavailable, SessionExpired) as e:
                retry_count += 1
                if retry_count == self.max_retries:
                    raise Exception(
                        f"Failed to execute query after {self.max_retries} attempts: {str(e)}"
                    )
                time.sleep(1)  # Wait 1 second before retrying
                self._connect()  # Try to reconnect