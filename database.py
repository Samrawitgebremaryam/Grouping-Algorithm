from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired
from typing import Dict, Any
import time
import nanoid


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

    def get_graph_data(self, request_data: Dict, limit: int = 1000) -> Dict[str, Any]:
        try:
            with self.driver.session() as session:
                # First, let's check what labels exist in the database
                label_query = """
                CALL db.labels() YIELD label 
                RETURN collect(toLower(label)) as labels
                """
                labels_result = session.run(label_query).single()
                available_labels = labels_result["labels"] if labels_result else []
                print(f"Available labels in database: {available_labels}")

                # Get relationship types
                rel_query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN collect(toLower(relationshipType)) as types
                """
                rel_result = session.run(rel_query).single()
                available_rels = rel_result["types"] if rel_result else []
                print(f"Available relationship types: {available_rels}")

                # Get properties from request for each node type
                node_properties = {}
                for node in request_data.get("nodes", []):
                    node_type = node.get("type", "").lower()
                    properties = node.get("properties", {})
                    if node_type:
                        node_properties[node_type] = properties

                print(f"Node properties to match: {node_properties}")

                # Build a case-insensitive query
                query = """
                MATCH (g)
                WHERE any(label IN labels(g) WHERE toLower(label) = 'gene')
                AND toLower(g.gene_name) = toLower($gene_name)
                WITH g
                MATCH (g)-[r]->(t)
                WHERE any(label IN labels(t) WHERE toLower(label) = 'transcript')
                RETURN 
                    collect(distinct g) as nodes,
                    collect(distinct t) as transcripts,
                    collect(distinct {rel: r, start: g, end: t}) as relationships
                """

                # Get gene name from properties
                gene_properties = node_properties.get("gene", {})
                gene_name = gene_properties.get("gene_name")

                print(f"Querying for gene: {gene_name}")

                result = session.run(query, gene_name=gene_name)
                record = result.single()

                if not record:
                    print("No records returned from database")
                    return {"nodes": [], "edges": []}

                nodes = []
                edges = []

                # Process gene nodes
                for node in record["nodes"]:
                    try:
                        node_properties = dict(node.items())
                        node_id = node_properties.get("id", str(node.id))
                        node_data = {
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
                        nodes.append({"data": node_data})
                    except Exception as e:
                        print(f"Error processing gene node: {str(e)}")
                        continue

                # Process transcript nodes
                for node in record["transcripts"]:
                    try:
                        node_properties = dict(node.items())
                        node_id = node_properties.get("id", str(node.id))
                        node_data = {
                            "id": f"transcript {node_id}",
                            "type": "transcript",
                            "gene_name": gene_name,
                            "transcript_id": node_properties.get("transcript_id", ""),
                            "transcript_name": node_properties.get(
                                "transcript_name", ""
                            ),
                            "start": str(node_properties.get("start")),
                            "end": str(node_properties.get("end")),
                            "label": "transcript",
                            "transcript_type": node_properties.get(
                                "transcript_type", ""
                            ),
                            "chr": node_properties.get("chr"),
                            "name": f"transcript {node_id}",
                        }
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

                        if any(x is None for x in [rel, start_node, end_node]):
                            continue

                        start_id = start_node.get("id", str(start_node.id))
                        end_id = end_node.get("id", str(end_node.id))

                        edges.append(
                            {
                                "data": {
                                    "edge_id": f"gene_transcribed_to_transcript",
                                    "label": rel.type.lower(),
                                    "source": f"gene {start_id}",
                                    "target": f"transcript {end_id}",
                                    "id": f"e{nanoid.generate(size=10)}",
                                }
                            }
                        )
                    except Exception as e:
                        print(f"Error processing relationship: {str(e)}")
                        continue

                print(f"Final processed nodes: {len(nodes)}")
                print(f"Final processed edges: {len(edges)}")
                return {"nodes": nodes, "edges": edges}

        except (ServiceUnavailable, SessionExpired) as e:
            raise Exception(
                f"Failed to execute query after {self.max_retries} attempts: {str(e)}"
            )
