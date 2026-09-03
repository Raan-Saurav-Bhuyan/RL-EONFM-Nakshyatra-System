"""
Topology JSON Parser and Validator for SDM-EON Digital Twin.
Validates uploaded JSON files for network elements (ROADM nodes) and connections (fiber links).
"""

import json
import os
from typing import Tuple, Dict, Any, List

REQUIRED_ELEMENT_FIELDS = ["uid", "type"]
REQUIRED_CONNECTION_FIELDS = ["from_node", "to_node", "length"]

def validate_and_parse_topology(json_path: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validates a topology JSON file and parses it into a standard structure.

    Parameters
    ----------
    json_path : str
        Path to the JSON file to validate.

    Returns
    -------
    Tuple[bool, str, Dict[str, Any]]
        (is_valid, error_or_success_msg, parsed_graph_dict)
    """
    if not os.path.exists(json_path):
        return False, f"File does not exist: {json_path}", {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Invalid JSON format: {str(e)}", {}

    if not isinstance(data, dict):
        return False, "Top level structure must be a JSON object", {}

    if "elements" not in data or not isinstance(data["elements"], list):
        return False, "Missing or invalid 'elements' array", {}

    if "connections" not in data or not isinstance(data["connections"], list):
        return False, "Missing or invalid 'connections' array", {}

    # Validate elements (nodes): --->
    node_uids = set()
    for idx, elem in enumerate(data["elements"]):
        if not isinstance(elem, dict):
            return False, f"Element at index {idx} must be an object", {}
        for field in REQUIRED_ELEMENT_FIELDS:
            if field not in elem:
                return False, f"Element at index {idx} missing required field '{field}'", {}
        node_uids.add(str(elem["uid"]))

    # Validate connections (links): --->
    for idx, conn in enumerate(data["connections"]):
        if not isinstance(conn, dict):
            return False, f"Connection at index {idx} must be an object", {}
        for field in REQUIRED_CONNECTION_FIELDS:
            if field not in conn:
                return False, f"Connection at index {idx} missing required field '{field}'", {}

        from_node = str(conn["from_node"])
        to_node = str(conn["to_node"])
        if from_node not in node_uids:
            return False, f"Connection at index {idx} references unknown 'from_node': {from_node}", {}
        if to_node not in node_uids:
            return False, f"Connection at index {idx} references unknown 'to_node': {to_node}", {}

    # Format for graph visualizer: --->
    graph_data = {
        "nodes": [
            {
                "id": str(elem["uid"]),
                "label": f"ROADM {elem['uid']}",
                "type": elem.get("type", "ROADM")
            }
            for elem in data["elements"]
        ],
        "edges": [
            {
                "from": str(conn["from_node"]),
                "to": str(conn["to_node"]),
                "length": float(conn.get("length", 1000.0)),
                "label": f"{conn.get('length', 1000.0)} km"
            }
            for conn in data["connections"]
        ],
        "num_nodes": len(data["elements"]),
        "num_edges": len(data["connections"])
    }

    return True, "Valid topology structure", graph_data
