"""
Headwise Linear Projection Pass

Transforms standard attention blocks by making the initial QKV linear
projections "head-wise". This avoids the expensive Reshape -> Transpose -> Split
sequence by splitting the QKV projection weights and creating smaller,
parallel linear layers per head.
"""

import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from collections import deque
from typing import List, Dict, Optional, Tuple

from ..base import OptimizationPass, Logger


class HeadwiseLinearPass(OptimizationPass):
    """
    Makes QKV linear projections head-wise.

    Detects attention blocks with:
    - LayerNorm -> QKV Linear -> Reshape -> Transpose -> Split
    - Per-head: MatMul_QK -> Softmax -> MatMul_PV
    - Concat -> ...

    Transforms to:
    - LayerNorm -> Per-head Q, K, V Linear
    - Per-head: MatMul_QK -> Softmax -> MatMul_PV
    - Concat (unchanged)
    """

    def __init__(self):
        super().__init__()
        self.block_id = 0

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header("Running Headwise Linear Projection Pass")

        self.model = model
        self._assign_unique_names()

        while True:
            # Reload maps for current state
            model_str = self.model.SerializeToString()
            self.model = onnx.load_model_from_string(model_str)
            self._build_maps()

            # Find attention blocks
            blocks = self._find_attention_blocks()

            if not blocks:
                Logger.success("No more attention blocks found. Pass complete.")
                break

            block = blocks[0]
            Logger.progress(
                f"Processing block {self.block_id}: {block['layernorm_node'].name}"
            )

            # Split initializers
            new_inits = self._split_qkv_initializers(block)
            Logger.info(f"Split into {block['num_heads']} head-wise chunks")

            # Reconstruct attention block
            self._reconstruct_attention_block(block, new_inits)
            Logger.success(f"Block {self.block_id} transformed")

            # Remove duplicate initializers
            self._deduplicate_initializers()

        # Final shape inference
        try:
            graph_input_names = {i.name for i in self.model.graph.input}
            inputs_vi = [
                vi for vi in self.model.graph.value_info
                if vi.name in graph_input_names
            ]
            self.model.graph.ClearField("value_info")
            self.model.graph.value_info.extend(inputs_vi)
            self._topologically_sort_nodes()
            self.model = onnx.shape_inference.infer_shapes(self.model)
            Logger.success("Shape inference complete")
        except Exception as e:
            Logger.warning(f"Shape inference warning: {e}")

        return self.model

    def _find_attention_blocks(self) -> List[Dict]:
        """Find attention blocks that can be transformed."""
        blocks = []

        for node in self.model.graph.node:
            if node.op_type != 'Concat':
                continue

            num_heads = len(node.input)
            if num_heads < 3:
                continue

            try:
                block = self._trace_attention_block(node, num_heads)
                if block:
                    blocks.append(block)
                    self.block_id += 1
            except (KeyError, IndexError, ValueError):
                continue

        return blocks

    def _trace_attention_block(
        self, concat_node: onnx.NodeProto, num_heads: int
    ) -> Optional[Dict]:
        """Trace back from Concat to find the complete attention block."""
        per_head_nodes = {}
        all_q_splits = []
        all_k_splits = []
        all_v_splits = []

        for i in range(num_heads):
            # Trace back from Concat input
            pv_matmul = self.producer_map.get(concat_node.input[i])
            if not pv_matmul or pv_matmul.op_type != 'MatMul':
                return None

            # Find Softmax
            softmax, _ = self._trace_back_to_node(pv_matmul, 'Softmax')
            if not softmax:
                return None

            # Find QK MatMul
            qk_matmul, sf_to_qk_path = self._trace_back_to_node(softmax, 'MatMul')
            if not qk_matmul:
                return None

            # Find Q and K Split nodes
            scale_q_mul = None
            q_producer = self.producer_map.get(qk_matmul.input[0])
            if q_producer and q_producer.op_type == 'Mul':
                scale_q_mul = q_producer
                q_split = self.producer_map.get(scale_q_mul.input[0])
            else:
                q_split = q_producer

            scale_k_mul = None
            k_producer = self.producer_map.get(qk_matmul.input[1])
            if k_producer and k_producer.op_type == 'Mul':
                scale_k_mul = k_producer
                k_split = self.producer_map.get(scale_k_mul.input[0])
            else:
                k_split = k_producer

            # Find V Split node
            v_split = self.producer_map.get(pv_matmul.input[1])

            if not q_split or not k_split or not v_split:
                return None

            per_head_nodes[i] = {
                'pv_matmul': pv_matmul,
                'softmax': softmax,
                'qk_matmul': qk_matmul,
                'scale_q_mul': scale_q_mul,
                'scale_k_mul': scale_k_mul,
                'sf_to_qk_path': sf_to_qk_path
            }
            all_q_splits.append(q_split)
            all_k_splits.append(k_split)
            all_v_splits.append(v_split)

        # Validate all splits are Split nodes
        if not all(s.op_type == 'Split' for s in all_q_splits):
            return None
        if not all(s.op_type == 'Split' for s in all_k_splits):
            return None
        if not all(s.op_type == 'Split' for s in all_v_splits):
            return None

        # Trace back to find linear layers
        q_linear = self._trace_back_to_linear(all_q_splits[0])
        k_linear = self._trace_back_to_linear(all_k_splits[0])
        v_linear = self._trace_back_to_linear(all_v_splits[0])

        if not q_linear or not k_linear or not v_linear:
            return None

        # Verify common input
        common_input = q_linear['matmul'].input[0]
        if common_input != k_linear['matmul'].input[0]:
            return None
        if common_input != v_linear['matmul'].input[0]:
            return None

        # Find input producer (LayerNorm or other)
        input_producer = self.producer_map.get(common_input)
        if not input_producer:
            return None

        # Collect nodes to delete. DA3's no-RoPE blocks feed K into the QK
        # MatMul through one extra transpose compared with DAV2:
        #   Transpose_k([B,H,N,D]) -> Transpose_1([B,H,D,N]) -> Split_K
        # Head-wise linear reconstruction creates the per-head K transpose
        # directly, so the extra transpose must be removed as part of the
        # original projection/layout chain.
        nodes_to_delete = [
            q_linear['matmul'], q_linear['add'],
            q_linear['reshape'], q_linear['transpose'], all_q_splits[0],
            k_linear['matmul'], k_linear['add'],
            k_linear['reshape'], k_linear['transpose'], all_k_splits[0],
            v_linear['matmul'], v_linear['add'],
            v_linear['reshape'], v_linear['transpose'], all_v_splits[0],
        ]
        if k_linear.get('post_transpose') is not None:
            nodes_to_delete.append(k_linear['post_transpose'])
        unique_nodes = {n.name: n for n in nodes_to_delete}

        return {
            "layernorm_node": input_producer,
            "layernorm_output": input_producer.output[0],
            "num_heads": num_heads,
            "q_linear": q_linear,
            "k_linear": k_linear,
            "v_linear": v_linear,
            "nodes_to_delete": list(unique_nodes.values()),
            "per_head_nodes": per_head_nodes,
            "final_concat": concat_node,
            "block_id": self.block_id
        }

    def _trace_back_to_node(
        self,
        start_node: onnx.NodeProto,
        target_type: str
    ) -> Tuple[Optional[onnx.NodeProto], List[onnx.NodeProto]]:
        """Trace backwards to find a target node type."""
        TRANSPARENT_OPS = ['Add', 'Div', 'Mul', 'Cast', 'Reshape', 'Transpose']

        queue = deque([(start_node, [])])
        visited = {start_node.name}

        while queue:
            curr, path = queue.popleft()
            producer = self.producer_map.get(curr.input[0])

            if not producer or producer.name in visited:
                continue

            if producer.op_type == target_type:
                return producer, path[::-1]

            if producer.op_type in TRANSPARENT_OPS:
                visited.add(producer.name)
                queue.append((producer, [producer] + path))

        return None, []

    def _trace_back_to_linear(
        self, split_node: onnx.NodeProto
    ) -> Optional[Dict]:
        """Trace from Split back to find the linear layer."""
        transpose = self.producer_map.get(split_node.input[0])
        if not transpose or transpose.op_type != 'Transpose':
            return None

        post_transpose = None
        reshape = self.producer_map.get(transpose.input[0])
        if reshape and reshape.op_type == 'Transpose':
            post_transpose = transpose
            transpose = reshape
            reshape = self.producer_map.get(transpose.input[0])
        if not reshape or reshape.op_type != 'Reshape':
            return None

        add = self.producer_map.get(reshape.input[0])
        if not add or add.op_type != 'Add':
            return None

        # MatMul could be either input to Add
        matmul = self.producer_map.get(add.input[0])
        if not matmul or matmul.op_type != 'MatMul':
            matmul = self.producer_map.get(add.input[1])
            if not matmul or matmul.op_type != 'MatMul':
                return None

        return {
            "transpose": transpose,
            "post_transpose": post_transpose,
            "reshape": reshape,
            "add": add,
            "matmul": matmul
        }

    def _split_qkv_initializers(self, block: Dict) -> Dict:
        """Split QKV weights and biases into per-head chunks."""
        num_heads = block['num_heads']
        all_new_inits = {}

        for branch in ['q', 'k', 'v']:
            linear = block[f"{branch}_linear"]

            # Find weight and bias
            weight_name = next(
                inp for inp in linear['matmul'].input
                if inp in self.initializer_map
            )
            bias_name = next(
                inp for inp in linear['add'].input
                if inp in self.initializer_map
            )

            weight_np = numpy_helper.to_array(self.initializer_map[weight_name])
            bias_np = numpy_helper.to_array(self.initializer_map[bias_name])

            # Split along output dimension
            split_weights = np.split(weight_np, num_heads, axis=1)
            split_biases = np.split(bias_np, num_heads, axis=0)

            new_weights = []
            new_biases = []

            for i in range(num_heads):
                w_init = numpy_helper.from_array(
                    split_weights[i], name=f"{weight_name}_head_{i}"
                )
                b_init = numpy_helper.from_array(
                    split_biases[i], name=f"{bias_name}_head_{i}"
                )
                new_weights.append(w_init)
                new_biases.append(b_init)

            all_new_inits[f'{branch}_weights'] = new_weights
            all_new_inits[f'{branch}_biases'] = new_biases

        return all_new_inits

    def _reconstruct_attention_block(self, block: Dict, new_inits: Dict):
        """Reconstruct the attention block with head-wise linear layers."""
        num_heads = block['num_heads']
        block_id = block['block_id']
        ln_output = block['layernorm_output']
        new_nodes = []

        # Fix attention bias shapes if needed
        for head_nodes in block['per_head_nodes'].values():
            if head_nodes['sf_to_qk_path']:
                for node in head_nodes['sf_to_qk_path']:
                    if node.op_type == 'Add' and node.input[1] in self.initializer_map:
                        bias_name = node.input[1]
                        bias_init = self.initializer_map[bias_name]
                        bias_np = numpy_helper.to_array(bias_init)
                        if len(bias_np.shape) == 4:
                            squeezed = np.squeeze(bias_np, axis=1)
                            new_init = numpy_helper.from_array(squeezed, name=bias_name)
                            for i, init in enumerate(self.model.graph.initializer):
                                if init.name == bias_name:
                                    self.model.graph.initializer[i].CopyFrom(new_init)
                                    break
                            self.initializer_map[bias_name] = new_init

        # Add new initializers
        for branch in ['q', 'k', 'v']:
            self.model.graph.initializer.extend(new_inits[f'{branch}_weights'])
            self.model.graph.initializer.extend(new_inits[f'{branch}_biases'])

        # Create per-head linear layers and rewire
        per_head_outputs = []

        for i in range(num_heads):
            branch_outputs = {}

            # Create Q, K, V linear projections
            for branch in ['q', 'k', 'v']:
                weight = new_inits[f'{branch}_weights'][i]
                bias = new_inits[f'{branch}_biases'][i]

                mm_out = f"{ln_output}_{branch}_matmul_head_{i}"
                add_out = f"{ln_output}_{branch}_add_head_{i}"

                mm_node = helper.make_node(
                    'MatMul',
                    inputs=[ln_output, weight.name],
                    outputs=[mm_out],
                    name=f"block{block_id}_MatMul_{branch}_head_{i}"
                )
                add_node = helper.make_node(
                    'Add',
                    inputs=[mm_out, bias.name],
                    outputs=[add_out],
                    name=f"block{block_id}_Add_{branch}_head_{i}"
                )
                new_nodes.extend([mm_node, add_node])
                branch_outputs[branch] = add_out

            # Transpose K
            k_transposed = f"{branch_outputs['k']}_transposed"
            transpose_k = helper.make_node(
                'Transpose',
                inputs=[branch_outputs['k']],
                outputs=[k_transposed],
                name=f"block{block_id}_Transpose_k_head_{i}",
                perm=[0, 2, 1]
            )
            new_nodes.append(transpose_k)

            # Rewire existing per-head nodes
            head_nodes = block['per_head_nodes'][i]
            q_input = branch_outputs['q']
            k_input = k_transposed

            # Handle scale multiplication
            if head_nodes.get('scale_q_mul'):
                head_nodes['scale_q_mul'].input[0] = branch_outputs['q']
                self._retarget_scale_shape(head_nodes['scale_q_mul'].input[1], branch_outputs['q'])
                q_input = head_nodes['scale_q_mul'].output[0]

            if head_nodes.get('scale_k_mul'):
                head_nodes['scale_k_mul'].input[0] = k_transposed
                self._retarget_scale_shape(head_nodes['scale_k_mul'].input[1], branch_outputs['q'])
                k_input = head_nodes['scale_k_mul'].output[0]

            # Rewire QK and PV MatMuls
            head_nodes['qk_matmul'].input[0] = q_input
            head_nodes['qk_matmul'].input[1] = k_input
            head_nodes['pv_matmul'].input[1] = branch_outputs['v']

            per_head_outputs.append(head_nodes['pv_matmul'].output[0])

        # Create Unsqueeze nodes for proper Concat
        unsqueeze_nodes = []
        new_concat_inputs = []

        for i in range(num_heads):
            axes_name = f"unsqueeze_axes_head_{i}"
            axes_init = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=[1],
                vals=[1]
            )
            self.model.graph.initializer.append(axes_init)

            unsqueezed = f"{per_head_outputs[i]}_unsqueezed"
            unsqueeze = helper.make_node(
                'Unsqueeze',
                inputs=[per_head_outputs[i], axes_name],
                outputs=[unsqueezed],
                name=f"{per_head_outputs[i]}_Unsqueeze_head_{i}"
            )
            unsqueeze_nodes.append(unsqueeze)
            new_concat_inputs.append(unsqueezed)

        # Rewire Concat
        concat = block['final_concat']
        concat.ClearField('input')
        concat.input.extend(new_concat_inputs)

        # Build final node list
        nodes_to_delete = {n.name for n in block['nodes_to_delete']}
        unsqueeze_map = {n.input[0]: n for n in unsqueeze_nodes}
        final_nodes = []
        linear_inserted = False

        for node in self.model.graph.node:
            if node.name in nodes_to_delete:
                continue

            final_nodes.append(node)

            # Insert linear layers after input producer
            if node.name == block['layernorm_node'].name and not linear_inserted:
                final_nodes.extend(new_nodes)
                linear_inserted = True

            # Insert Unsqueeze after PV MatMul
            for out in node.output:
                if out in unsqueeze_map:
                    final_nodes.append(unsqueeze_map[out])

        self.model.graph.ClearField('node')
        self.model.graph.node.extend(final_nodes)

        # Remove old initializers
        old_inits = set()
        for branch in ['q', 'k', 'v']:
            linear = block[f'{branch}_linear']
            old_inits.add(next(
                i for i in linear['matmul'].input if i in self.initializer_map
            ))
            old_inits.add(next(
                i for i in linear['add'].input if i in self.initializer_map
            ))

        final_inits = [
            init for init in self.model.graph.initializer
            if init.name not in old_inits
        ]
        self.model.graph.ClearField('initializer')
        self.model.graph.initializer.extend(final_inits)

    def _retarget_scale_shape(self, scale_tensor: str, replacement_tensor: str) -> None:
        """Point dynamic attention-scale Shape nodes at the new per-head tensor."""
        queue = deque([scale_tensor])
        visited = set()

        while queue:
            tensor = queue.popleft()
            if tensor in visited:
                continue
            visited.add(tensor)

            producer = self.producer_map.get(tensor)
            if producer is None:
                continue

            if producer.op_type == 'Shape':
                producer.input[0] = replacement_tensor
                continue

            for inp in producer.input:
                if inp in self.producer_map:
                    queue.append(inp)

    def _deduplicate_initializers(self):
        """Remove duplicate initializers."""
        seen = {}
        for init in self.model.graph.initializer:
            if init.name not in seen:
                seen[init.name] = init

        if len(seen) < len(self.model.graph.initializer):
            removed = len(self.model.graph.initializer) - len(seen)
            Logger.info(f"Removed {removed} duplicate initializers")
            self.model.graph.ClearField("initializer")
            self.model.graph.initializer.extend(seen.values())
