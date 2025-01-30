from flask import Flask, request, jsonify, stream_with_context, Response
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
    neo4j_conn = None
    try:
        request_json = request.json
        print("Full request data:", json.dumps(request_json, indent=2))

        # Initialize Neo4j connection
        neo4j_conn = Neo4jConnection(
            uri="neo4j+s://cb5ba820.databases.neo4j.io",
            user="neo4j",
            password="EiegSoXk1JX8vviqSOPpf9czT_lmX9mixd_b9rn0VrQ",
            max_retries=3,
        )

        # Get request data and pagination parameters
        request_data = request_json.get("requests", {})
        limit = request_json.get("limit", 1000)

        print(f"Processing request with limit: {limit}")
        print(f"Request data: {json.dumps(request_data, indent=2)}")

        if not request_data:
            return jsonify({"error": "No requests data provided"}), 400

        # Get data from Neo4j with limit - Pass request_data here
        graph_data = neo4j_conn.get_graph_data(request_data=request_data, limit=limit)
        print(f"Raw graph data from Neo4j: {json.dumps(graph_data, indent=2)}")

        # Create graph instance using Neo4j data
        input_graph = Graph.from_dict(graph_data)
        print(
            f"Created graph instance - Nodes: {len(input_graph.nodes)}, Edges: {len(input_graph.edges)}"
        )

        # Process the graph with the requests data
        grouped_graph = group_graph(input_graph, request_data)
        print(
            f"Processed grouped graph - Nodes: {len(grouped_graph.nodes)}, Edges: {len(grouped_graph.edges)}"
        )

        # Convert to dictionary
        result_dict = grouped_graph.to_dict()

        # Stream the response
        def generate():
            yield '{"nodes": ['
            for i, node in enumerate(result_dict["nodes"]):
                if i > 0:
                    yield ","
                yield json.dumps(node)
            yield '], "edges": ['
            for i, edge in enumerate(result_dict["edges"]):
                if i > 0:
                    yield ","
                yield json.dumps(edge)
            yield "]}"

        return Response(stream_with_context(generate()), mimetype="application/json")

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500
    finally:
        if neo4j_conn:
            neo4j_conn.close()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
