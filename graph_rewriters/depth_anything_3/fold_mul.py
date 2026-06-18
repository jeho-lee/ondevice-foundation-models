from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from ..base import Logger, OptimizationPass


class _Da3MulFoldBase(OptimizationPass):
    def _refresh(self) -> None:
        self._build_maps()
        self.initializers = {
            init.name: numpy_helper.to_array(init).copy()
            for init in self.model.graph.initializer
        }

    def _set_initializer(self, name: str, value: np.ndarray) -> None:
        for idx, init in enumerate(self.model.graph.initializer):
            if init.name == name:
                self.model.graph.initializer.remove(init)
                self.model.graph.initializer.insert(idx, numpy_helper.from_array(value.astype(self.initializers[name].dtype), name))
                self.initializers[name] = value.astype(self.initializers[name].dtype)
                return
        raise KeyError(name)

    def _remove_nodes(self, remove_names: set[str]) -> None:
        keep = [node for node in self.model.graph.node if node.name not in remove_names]
        self.model.graph.ClearField("node")
        self.model.graph.node.extend(keep)

    def _replace_input(self, old: str, new: str) -> None:
        for node in self.model.graph.node:
            for idx, input_name in enumerate(node.input):
                if input_name == old:
                    node.input[idx] = new
        for output in self.model.graph.output:
            if output.name == old:
                output.name = new

    def _fold_linear_mul(self, mul: onnx.NodeProto) -> bool:
        if mul.op_type != "Mul" or len(mul.input) != 2:
            return False

        scale_name = None
        value_name = None
        for input_name in mul.input:
            if input_name in self.initializers:
                scale_name = input_name
            else:
                value_name = input_name
        if not scale_name or not value_name:
            return False

        add = self.producer_map.get(value_name)
        if not add or add.op_type != "Add":
            return False
        if self.consumer_map.get(add.output[0], []) != [mul]:
            return False

        matmul = None
        bias_name = None
        for input_name in add.input:
            producer = self.producer_map.get(input_name)
            if producer and producer.op_type == "MatMul":
                matmul = producer
            elif input_name in self.initializers:
                bias_name = input_name
        if not matmul or not bias_name or len(matmul.input) < 2:
            return False
        weight_name = matmul.input[1]
        if weight_name not in self.initializers:
            return False

        scale = np.asarray(self.initializers[scale_name])
        weight = np.asarray(self.initializers[weight_name])
        bias = np.asarray(self.initializers[bias_name])
        if scale.size == 1:
            scale_for_weight = scale.reshape(())
            scale_for_bias = scale.reshape(())
        elif weight.ndim == 2 and weight.shape[-1] == scale.size and bias.shape[-1] == scale.size:
            scale_for_weight = scale.reshape(1, -1)
            scale_for_bias = scale.reshape(-1)
        else:
            return False

        self._set_initializer(weight_name, weight * scale_for_weight)
        self._set_initializer(bias_name, bias * scale_for_bias)
        self._replace_input(mul.output[0], add.output[0])
        return True


class LayerScaleFoldPass(_Da3MulFoldBase):
    """Fold DA3 LayerScale Mul into the preceding projection weight and bias."""

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header("Running DA3 LayerScale Fold Pass")
        self.model = model
        folded: list[str] = []
        self._refresh()

        for node in list(self.model.graph.node):
            if node.op_type != "Mul" or "/ls" not in node.name:
                continue
            if self._fold_linear_mul(node):
                folded.append(node.name)
                self._refresh()

        self._remove_nodes(set(folded))
        Logger.success(f"LayerScale Fold complete: {len(folded)} Mul node(s)")
        return self.model
