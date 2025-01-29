from flask import Flask, request, jsonify
from neo4j import GraphDatabase
from dataclasses import dataclass
from typing import Dict, List, Any
import json

# Import your existing classes and functions
from index import Graph, Node, Edge, group_graph, count_nodes_by_type
from database import Neo4jConnection

app = Flask(__name__)


@app.route("/api/graph", methods=["POST"])
def process_graph():
    try:
        # Initialize Neo4j connection
        neo4j_conn = Neo4jConnection(
            uri="neo4j+s://cb5ba820.databases.neo4j.io",
            user="neo4j",
            password="EiegSoXk1JX8vviqSOPpf9czT_lmX9mixd_b9rn0VrQ",
        )

        # Get request data from Postman and extract the 'requests' object
        request_data = request.json.get("requests", {})
        if not request_data:
            return jsonify({"error": "No requests data provided"}), 400

        try:
            # Get data from Neo4j
            graph_data = neo4j_conn.get_graph_data()

            # Create graph instance using Neo4j data
            input_graph = Graph.from_dict(graph_data)

            # Process the graph with the requests data
            grouped_graph = group_graph(input_graph, request_data)

            # Return just the nodes and edges directly
            return jsonify(grouped_graph.to_dict())

        finally:
            neo4j_conn.close()

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
