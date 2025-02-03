from flask import Flask, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
import json
import os
from functools import wraps

# Import from GroupingAlgorithm.py
from GroupingAlgorithm import Graph, Neo4jConnection, group_graph

# Load environment variables
load_dotenv()

app = Flask(__name__)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # Add your token validation logic here if needed
        return f(*args, **kwargs)

    return decorated


@app.route("/api/graph", methods=["POST"])
@token_required
def process_graph(current_user_id=None):
    neo4j_conn = None
    try:
        request_json = request.json
        print("Full request data:", json.dumps(request_json, indent=2))

        # Initialize Neo4j connection
        neo4j_conn = Neo4jConnection(
            uri=os.getenv("NEO4J_URI"),
            user=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            max_retries=int(os.getenv("NEO4J_MAX_RETRIES", 3)),
        )

        # Get request data and pagination parameters
        request_data = request_json.get("requests", {})
        limit = request_json.get("limit", 1000)

        print(f"Processing request with limit: {limit}")
        print(f"Request data: {json.dumps(request_data, indent=2)}")

        if not request_data:
            return jsonify({"error": "No requests data provided"}), 400

        # Get data from Neo4j with limit
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
                # Ensure node IDs include node type
                node_data = node["data"]
                if (
                    "type" in node_data
                    and "id" in node_data
                    and not node_data["id"].startswith(node_data["type"])
                ):
                    node_data["id"] = f"{node_data['type']} {node_data['id']}"
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
