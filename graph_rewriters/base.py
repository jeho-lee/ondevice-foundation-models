"""
Base classes and utilities for optimization passes.
"""

import onnx
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


# ANSI colors
class Colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


class Logger:
    """Simple logging utility."""

    @staticmethod
    def header(text: str):
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD}  {text}{Colors.RESET}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.RESET}")

    @staticmethod
    def info(text: str):
        print(f"{Colors.INFO}[INFO]{Colors.RESET} {text}")

    @staticmethod
    def success(text: str):
        print(f"{Colors.SUCCESS}[OK]{Colors.RESET} {text}")

    @staticmethod
    def warning(text: str):
        print(f"{Colors.WARNING}[WARN]{Colors.RESET} {text}")

    @staticmethod
    def error(text: str):
        print(f"{Colors.ERROR}[ERROR]{Colors.RESET} {text}")

    @staticmethod
    def progress(text: str):
        print(f"{Colors.INFO}  -> {text}{Colors.RESET}")

    @staticmethod
    def debug(text: str):
        print(f"{Colors.WARNING}[DEBUG]{Colors.RESET} {text}")


class OptimizationPass(ABC):
    """
    Abstract base class for all optimization passes.

    Each pass takes an ONNX model, modifies it, and returns the modified model.
    Subclasses must implement the `run` method.
    """

    def __init__(self):
        self.model: Optional[onnx.ModelProto] = None
        self.producer_map: Dict[str, onnx.NodeProto] = {}
        self.consumer_map: Dict[str, List[onnx.NodeProto]] = {}
        self.initializer_map: Dict[str, onnx.TensorProto] = {}
        self.value_info_map: Dict[str, onnx.ValueInfoProto] = {}

    @abstractmethod
    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply the optimization pass to the model.

        Args:
            model: Input ONNX model

        Returns:
            Modified ONNX model
        """
        pass

    def _build_maps(self):
        """
        Build producer/consumer/initializer maps for the graph.

        These maps are essential for graph traversal and modification.
        """
        self.producer_map = {}
        self.consumer_map = defaultdict(list)

        for node in self.model.graph.node:
            for output_name in node.output:
                self.producer_map[output_name] = node
            for input_name in node.input:
                self.consumer_map[input_name].append(node)

        self.initializer_map = {
            init.name: init for init in self.model.graph.initializer
        }

        # Build value_info map (includes inputs, outputs, and intermediates)
        self.value_info_map = {
            vi.name: vi for vi in self.model.graph.value_info
        }
        self.value_info_map.update({
            i.name: i for i in self.model.graph.input
        })
        self.value_info_map.update({
            o.name: o for o in self.model.graph.output
        })

    def _get_tensor_shape(self, tensor_name: str) -> Optional[List[int]]:
        """Get the shape of a tensor from value_info."""
        vi = self.value_info_map.get(tensor_name)
        if vi and vi.type.tensor_type.HasField('shape'):
            return [
                dim.dim_value for dim in vi.type.tensor_type.shape.dim
            ]
        return None

    def _find_transpose_ancestor(
        self, start_node: onnx.NodeProto
    ) -> Tuple[Optional[onnx.NodeProto], Optional[onnx.NodeProto]]:
        """
        Trace back from a node to find a Transpose ancestor.

        Handles two cases:
        1. Node -> Transpose (direct connection)
        2. Node -> Mul -> Transpose (scaled connection)

        Returns:
            Tuple of (transpose_node, mul_node or None)
        """
        if start_node.op_type == 'Transpose':
            return (start_node, None)

        if start_node.op_type == "Mul":
            parent_tensor = start_node.input[0]
            if parent_tensor in self.producer_map:
                parent_node = self.producer_map[parent_tensor]
                if parent_node.op_type == "Transpose":
                    return (parent_node, start_node)

        if start_node.op_type == "Reshape":
            parent_tensor = start_node.input[0]
            if parent_tensor in self.producer_map:
                parent_node = self.producer_map[parent_tensor]
                if parent_node.op_type == "Transpose":
                    return (parent_node, start_node)

        return (None, None)

    def _find_v_transpose_and_softmax(
        self, matmul_qk: onnx.NodeProto
    ) -> Optional[Tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]]:
        """
        Find the V Transpose node by tracing forward from QK MatMul.

        Returns:
            Tuple of (transpose_v, softmax_node, matmul_pv) or None
        """
        scores_tensor = matmul_qk.output[0]
        consumers = self.consumer_map.get(scores_tensor, [])

        # Find Softmax
        softmax_node = None
        for consumer in consumers:
            if consumer.op_type == "Softmax":
                softmax_node = consumer
                break

        if not softmax_node:
            return None

        # Find PV MatMul
        probs_tensor = softmax_node.output[0]
        consumers = self.consumer_map.get(probs_tensor, [])

        matmul_pv = None
        for consumer in consumers:
            if consumer.op_type == "MatMul":
                matmul_pv = consumer
                break

        if not matmul_pv:
            return None

        # Find V tensor (the input to PV MatMul that's not probs)
        v_tensor = None
        for inp in matmul_pv.input:
            if inp != probs_tensor:
                v_tensor = inp
                break

        if not v_tensor:
            return None

        # Get V's producer (should be Transpose)
        transpose_v = self.producer_map.get(v_tensor)
        if transpose_v and transpose_v.op_type == "Transpose":
            return (transpose_v, softmax_node, matmul_pv)

        return None

    def _topologically_sort_nodes(self):
        """
        Sort nodes in topological order.

        This ensures that nodes are ordered such that each node comes after
        all nodes that produce its inputs.
        """
        graph = self.model.graph
        nodes = list(graph.node)
        node_by_name = {n.name: n for n in nodes}

        # Build adjacency and in-degree
        produced_by = {o: n.name for n in nodes for o in n.output}
        adj = {n.name: [] for n in nodes}
        in_degree = {n.name: 0 for n in nodes}

        for n in nodes:
            for inp in n.input:
                if inp and inp in produced_by:
                    p_name = produced_by[inp]
                    if p_name != n.name:
                        adj[p_name].append(n.name)
                        in_degree[n.name] += 1

        # Kahn's algorithm
        queue = deque([
            name for name, degree in in_degree.items() if degree == 0
        ])
        sorted_nodes = []

        while queue:
            u_name = queue.popleft()
            sorted_nodes.append(node_by_name[u_name])
            for v_name in adj[u_name]:
                in_degree[v_name] -= 1
                if in_degree[v_name] == 0:
                    queue.append(v_name)

        if len(sorted_nodes) == len(nodes):
            graph.ClearField("node")
            graph.node.extend(sorted_nodes)
        else:
            Logger.warning(
                f"Topological sort incomplete: {len(sorted_nodes)}/{len(nodes)} nodes"
            )

    def _assign_unique_names(self):
        """Assign unique names to unnamed nodes."""
        for i, node in enumerate(self.model.graph.node):
            if not node.name:
                node.name = f"Unnamed_{node.op_type}_{i}"
