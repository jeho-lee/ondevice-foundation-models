from __future__ import annotations

import re
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..base import Logger, OptimizationPass


class PreQkvExplosionPass(OptimizationPass):
    """Explode DA3 fused QKV projection while preserving the 4D head layout.

    DA3 exports DINO-style attention as:
      MatMul -> Add -> Reshape[B,N,3,H,Dh] -> Transpose[2,B,H,N,Dh]
        -> Gather(Q/K/V)

    Generic QKV explosion can accidentally collapse some paths to 3D tensors,
    which breaks DA3 blocks that contain q_norm/k_norm/RoPE. This pass only
    removes the fused QKV and layout split, and reconnects each branch with the
    exact shape that the original Gather nodes produced: [B,H,N,Dh].
    """

    def __init__(self, include_blocks: set[int] | None = None, label: str = "all"):
        super().__init__()
        self.include_blocks = include_blocks
        self.label = label

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header(f"Running DA3 4D-preserving Pre-QKV Explosion Pass ({self.label})")
        self.model = model
        processed = 0

        while True:
            self.initializers = {init.name: init for init in self.model.graph.initializer}
            self._build_maps()
            pattern = self._find_next_pattern()
            if not pattern:
                break
            self._apply_pattern(pattern)
            processed += 1

        Logger.success(f"DA3 Pre-QKV Explosion complete: {processed} block(s)")
        return self.model

    def _find_next_pattern(self) -> Optional[dict]:
        for matmul in self.model.graph.node:
            if matmul.op_type != "MatMul" or not self._is_qkv_matmul(matmul):
                continue
            block_idx = self._block_index(matmul.name)
            if self.include_blocks is not None and block_idx not in self.include_blocks:
                continue
            if len(matmul.input) < 2 or matmul.input[1] not in self.initializers:
                continue
            weight = numpy_helper.to_array(self.initializers[matmul.input[1]])
            if weight.ndim != 2 or weight.shape[1] % 3 != 0:
                continue

            add = self._single_consumer(matmul.output[0], "Add")
            reshape = self._single_consumer(add.output[0], "Reshape") if add else None
            transpose = self._single_consumer(reshape.output[0], "Transpose") if reshape else None
            if not add or not reshape or not transpose:
                continue

            shape = self._shape_from_reshape(reshape)
            perm = self._perm(transpose)
            if not shape or len(shape) != 5 or shape[2] != 3 or perm != [2, 0, 3, 1, 4]:
                continue

            gathers = [
                node for node in self.consumer_map.get(transpose.output[0], [])
                if node.op_type == "Gather" and self._axis(node) == 0
            ]
            gathers = sorted(gathers, key=self._gather_index)
            if len(gathers) < 3 or [self._gather_index(g) for g in gathers[:3]] != [0, 1, 2]:
                continue

            bias_name = next((inp for inp in add.input if inp in self.initializers), None)
            if not bias_name:
                continue

            return {
                "matmul": matmul,
                "add": add,
                "reshape": reshape,
                "transpose": transpose,
                "gathers": gathers[:3],
                "shape": [int(x) for x in shape],
                "weight_name": matmul.input[1],
                "bias_name": bias_name,
            }
        return None

    def _apply_pattern(self, pattern: dict) -> None:
        matmul = pattern["matmul"]
        add = pattern["add"]
        reshape = pattern["reshape"]
        transpose = pattern["transpose"]
        gathers = pattern["gathers"]
        batch, seq_len, _, num_heads, head_dim = pattern["shape"]
        branch_shape = [batch, seq_len, num_heads, head_dim]

        weight = numpy_helper.to_array(self.initializers[pattern["weight_name"]])
        bias = numpy_helper.to_array(self.initializers[pattern["bias_name"]])
        weight_splits = np.split(weight, 3, axis=1)
        bias_splits = np.split(bias, 3, axis=0)

        input_tensor = matmul.input[0]
        new_nodes = []
        replacement = {}
        branch_names = ["q", "k", "v"]

        for idx, name in enumerate(branch_names):
            w_name = f"{pattern['weight_name']}_{name}"
            b_name = f"{pattern['bias_name']}_{name}"
            shape_name = f"{matmul.name}_{name}_shape"
            mm_out = f"{matmul.output[0]}_{name}_mm"
            add_out = f"{add.output[0]}_{name}"
            reshape_out = f"{reshape.output[0]}_{name}"
            branch_out = f"{transpose.output[0]}_{name}"

            self.model.graph.initializer.extend(
                [
                    numpy_helper.from_array(weight_splits[idx], name=w_name),
                    numpy_helper.from_array(bias_splits[idx], name=b_name),
                    helper.make_tensor(shape_name, TensorProto.INT64, [4], branch_shape),
                ]
            )
            new_nodes.extend(
                [
                    helper.make_node("MatMul", [input_tensor, w_name], [mm_out], name=f"{matmul.name}_{name}"),
                    helper.make_node("Add", [mm_out, b_name], [add_out], name=f"{add.name}_{name}"),
                    helper.make_node("Reshape", [add_out, shape_name], [reshape_out], name=f"{reshape.name}_{name}"),
                    helper.make_node(
                        "Transpose",
                        [reshape_out],
                        [branch_out],
                        name=f"{transpose.name}_{name}",
                        perm=[0, 2, 1, 3],
                    ),
                ]
            )
            replacement[gathers[idx].output[0]] = branch_out

        remove_names = {matmul.name, add.name, reshape.name, transpose.name, *(g.name for g in gathers)}
        insertion_producer = self.producer_map.get(input_tensor)
        final_nodes = []
        inserted = False
        for node in self.model.graph.node:
            if node.name in remove_names:
                continue
            for input_idx, input_name in enumerate(node.input):
                if input_name in replacement:
                    node.input[input_idx] = replacement[input_name]
            final_nodes.append(node)
            if insertion_producer and node.name == insertion_producer.name and not inserted:
                final_nodes.extend(new_nodes)
                inserted = True
        if not inserted:
            final_nodes = new_nodes + final_nodes

        self.model.graph.ClearField("node")
        self.model.graph.node.extend(final_nodes)

    def _is_qkv_matmul(self, node: onnx.NodeProto) -> bool:
        return bool(node.name and "blocks" in node.name and "attn" in node.name and "qkv" in node.name)

    @staticmethod
    def _block_index(node_name: str) -> int | None:
        match = re.search(r"blocks\.(\d+)", node_name)
        return int(match.group(1)) if match else None

    def _single_consumer(self, tensor: str, op_type: str) -> Optional[onnx.NodeProto]:
        consumers = self.consumer_map.get(tensor, [])
        if len(consumers) != 1 or consumers[0].op_type != op_type:
            return None
        return consumers[0]

    def _shape_from_reshape(self, node: onnx.NodeProto) -> Optional[list[int]]:
        if len(node.input) < 2:
            return None
        name = node.input[1]
        if name in self.initializers:
            return numpy_helper.to_array(self.initializers[name]).astype(np.int64).tolist()
        producer = self.producer_map.get(name)
        if producer and producer.op_type == "Constant" and producer.attribute:
            return numpy_helper.to_array(producer.attribute[0].t).astype(np.int64).tolist()
        return None

    @staticmethod
    def _perm(node: onnx.NodeProto) -> list[int]:
        for attr in node.attribute:
            if attr.name == "perm":
                return list(attr.ints)
        return []

    @staticmethod
    def _axis(node: onnx.NodeProto) -> int:
        for attr in node.attribute:
            if attr.name == "axis":
                return int(attr.i)
        return 0

    def _gather_index(self, node: onnx.NodeProto) -> int:
        if len(node.input) < 2:
            return 999
        name = node.input[1]
        if name in self.initializers:
            return int(numpy_helper.to_array(self.initializers[name]).item())
        producer = self.producer_map.get(name)
        if producer and producer.op_type == "Constant" and producer.attribute:
            return int(numpy_helper.to_array(producer.attribute[0].t).item())
        return 999


class PreQkvExplosionNoRopePass(PreQkvExplosionPass):
    """Apply DA3 QKV explosion only to blocks before q_norm/RoPE starts."""

    def __init__(self):
        super().__init__(include_blocks=set(range(4)), label="no_rope_blocks_0_3")
