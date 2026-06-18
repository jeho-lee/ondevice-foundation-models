"""
MHA Head-wise Split Pass

Finds Multi-Head Attention (MHA) blocks and splits them into parallel
per-head computations.

This enables fine-grained control over which heads use FP16 vs INT8
for mixed-precision quantization.
"""

import onnx
from onnx import helper, TensorProto
from typing import List, Dict, Optional, Tuple

from ..base import OptimizationPass, Logger


class MhaHeadwiseSplitPass(OptimizationPass):
    """
    Splits MHA blocks into parallel per-head computations.

    Detects attention patterns:
    - Transpose_Q, Transpose_K, Transpose_V (shape: [B, H, L, D])
    - QK MatMul -> Softmax -> PV MatMul

    Transforms to:
    - Split Q, K, V along head dimension
    - Per-head: MatMul_QK -> Softmax -> MatMul_PV
    - Concat outputs
    """

    def __init__(self, is_batch_head_swapped: bool = False, allow_rope: bool = False):
        super().__init__()
        self.is_batch_head_swapped = is_batch_head_swapped
        self.allow_rope = allow_rope

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header("Running MHA Head-wise Split Pass")

        self.model = model
        self.value_info = {vi.name: vi for vi in self.model.graph.value_info}
        block_index = 0

        while True:
            mha_patterns = self._find_attention_patterns()

            if not mha_patterns:
                Logger.success("No more MHA patterns found. Pass complete.")
                break

            mha_info = mha_patterns[0]
            Logger.progress(f"Processing block {block_index}: {mha_info['pattern']['QK_MatMul']}")

            # Generate split nodes
            split_nodes, head_tensors, new_inits = self._split_qkv_tensors(mha_info)

            # Replicate attention core for each head
            core_nodes, final_outputs = self._replicate_attention_core(mha_info, head_tensors)

            # Concat head outputs
            concat_node, final_name = self._concat_heads(mha_info, final_outputs)

            all_new_nodes = split_nodes + core_nodes + [concat_node]

            # Add initializers
            self.model.graph.initializer.extend(new_inits)

            # Rewire graph
            self._rewire_and_cleanup(mha_info, all_new_nodes, final_name)

            Logger.success(f"Block {block_index} split into {mha_info['num_heads']} heads")
            block_index += 1

        # Infer shapes
        self.model = onnx.shape_inference.infer_shapes(self.model)
        return self.model

    def _get_tensor_shape(self, tensor_name: str) -> Optional[List[int]]:
        """Get tensor shape from value_info."""
        vi = self.value_info.get(tensor_name)
        if vi:
            return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
        return None

    def _find_attention_patterns(self) -> List[Dict]:
        """Find all MHA patterns in the model."""
        self._build_maps()
        mha_infos = []

        for node in self.model.graph.node:
            if node.op_type != "MatMul" or len(node.input) != 2:
                continue

            # Check parents for Transpose (Q and K branches)
            parent_a = self.producer_map.get(node.input[0])
            parent_b = self.producer_map.get(node.input[1])

            if not parent_a or not parent_b:
                continue

            # Find Transpose ancestors
            transpose_q, mul_q = self._find_transpose_ancestor(parent_a)
            transpose_k, mul_k = self._find_transpose_ancestor(parent_b)

            # Try swapping if needed
            if not transpose_q or not transpose_k:
                transpose_k, mul_k = self._find_transpose_ancestor(parent_a)
                transpose_q, mul_q = self._find_transpose_ancestor(parent_b)

            if not transpose_q or not transpose_k:
                continue
            if transpose_q.name == transpose_k.name:
                continue

            # Find V branch
            result = self._find_v_transpose_and_softmax(node)
            if not result:
                continue

            transpose_v, softmax_node, matmul_pv = result

            # Get MHA info from Q shape
            shape_q = self._get_tensor_shape(transpose_q.output[0])
            if not shape_q or len(shape_q) != 4:
                continue

            if self.is_batch_head_swapped:
                num_heads, B, seq_len, head_dim = shape_q
            else:
                B, num_heads, seq_len, head_dim = shape_q

            pattern = {
                "QK_MatMul": node.name,
                "PV_MatMul": matmul_pv.name,
                "Softmax": softmax_node.name,
                "Transpose_Q": transpose_q.name,
                "Transpose_K": transpose_k.name,
                "Transpose_V": transpose_v.name,
                "Scale_Q_Mul": mul_q.name if mul_q else None,
                "Scale_K_Mul": mul_k.name if mul_k else None,
            }

            mha_infos.append({
                "num_heads": num_heads,
                "head_dim": head_dim,
                "tensor_to_split_q": transpose_q.output[0],
                "tensor_to_split_k": transpose_k.output[0],
                "tensor_to_split_v": transpose_v.output[0],
                "pattern": pattern
            })

        return mha_infos

    def _find_transpose_ancestor(
        self, start_node: onnx.NodeProto
    ) -> Tuple[Optional[onnx.NodeProto], Optional[onnx.NodeProto]]:
        transpose, mul = super()._find_transpose_ancestor(start_node)
        if transpose or not self.allow_rope:
            return transpose, mul

        # DA3 RoPE blocks feed Q into the attention MatMul as:
        #   rope/Concat_3 -> Mul(scale) -> MatMul(QK)
        # K still goes through an additional Transpose after rope_1/Concat_3.
        # Treat the RoPE concat as the head-layout tensor to split.
        if start_node.op_type == "Mul":
            parent = self.producer_map.get(start_node.input[0])
            if parent and parent.op_type == "Concat" and "/rope" in parent.name:
                return parent, start_node

        return None, None

    def _split_qkv_tensors(
        self, mha_info: Dict
    ) -> Tuple[List, Dict, List]:
        """Create Split nodes for Q, K, V tensors."""
        num_heads = mha_info['num_heads']
        pattern_name = mha_info["pattern"]["QK_MatMul"].replace("_QK_MatMul", "")

        new_nodes = []
        new_inits = []
        head_tensors = {"q": [], "k": [], "v": []}

        # Create split shape initializer
        split_name = f"{pattern_name}_split_info"
        split_init = helper.make_tensor(
            name=split_name,
            data_type=TensorProto.INT64,
            dims=[num_heads],
            vals=[1] * num_heads
        )
        new_inits.append(split_init)

        # Create Split nodes for Q, K, V
        for branch in ["q", "k", "v"]:
            input_tensor = mha_info[f"tensor_to_split_{branch}"]
            output_names = [
                f"{pattern_name}_{branch}_head_{i}" for i in range(num_heads)
            ]

            split_node = helper.make_node(
                "Split",
                inputs=[input_tensor, split_name],
                outputs=output_names,
                name=f"{pattern_name}_Split_{branch.upper()}",
                axis=0 if self.is_batch_head_swapped else 1
            )
            new_nodes.append(split_node)
            head_tensors[branch] = output_names

        return new_nodes, head_tensors, new_inits

    def _replicate_attention_core(
        self, mha_info: Dict, head_tensors: Dict
    ) -> Tuple[List, List]:
        """Create per-head attention computation."""
        num_heads = mha_info['num_heads']
        pattern = mha_info['pattern']
        pattern_name = pattern["QK_MatMul"].replace("_QK_MatMul", "")

        new_nodes = []
        final_outputs = []

        # Get scaling factors if present
        scalar_q = None
        if pattern["Scale_Q_Mul"]:
            mul_node = next(
                (n for n in self.model.graph.node if n.name == pattern["Scale_Q_Mul"]),
                None
            )
            if mul_node:
                q_out = mha_info["tensor_to_split_q"]
                scalar_q = next(
                    (inp for inp in mul_node.input if inp != q_out), None
                )

        scalar_k = None
        if pattern["Scale_K_Mul"]:
            mul_node = next(
                (n for n in self.model.graph.node if n.name == pattern["Scale_K_Mul"]),
                None
            )
            if mul_node:
                k_out = mha_info["tensor_to_split_k"]
                scalar_k = next(
                    (inp for inp in mul_node.input if inp != k_out), None
                )

        # Create per-head attention
        for i in range(num_heads):
            q_tensor = head_tensors['q'][i]
            k_tensor = head_tensors['k'][i]
            v_tensor = head_tensors['v'][i]

            q_input = q_tensor
            k_input = k_tensor

            # Scale Q if needed
            if scalar_q:
                scaled_q = f'{pattern_name}_scaled_q_head_{i}'
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[q_tensor, scalar_q],
                    outputs=[scaled_q],
                    name=f"{pattern_name}_Mul_Q_head_{i}"
                )
                new_nodes.append(mul_node)
                q_input = scaled_q

            # Scale K if needed
            if scalar_k:
                scaled_k = f'{pattern_name}_scaled_k_head_{i}'
                mul_node = helper.make_node(
                    "Mul",
                    inputs=[k_tensor, scalar_k],
                    outputs=[scaled_k],
                    name=f"{pattern_name}_Mul_K_head_{i}"
                )
                new_nodes.append(mul_node)
                k_input = scaled_k

            # QK MatMul
            scores = f"{pattern_name}_scores_head_{i}"
            qk_matmul = helper.make_node(
                "MatMul",
                inputs=[q_input, k_input],
                outputs=[scores],
                name=f"{pattern_name}_QK_MatMul_head_{i}"
            )
            new_nodes.append(qk_matmul)

            # Softmax
            probs = f"{pattern_name}_probs_head_{i}"
            softmax = helper.make_node(
                "Softmax",
                inputs=[scores],
                outputs=[probs],
                name=f"{pattern_name}_Softmax_head_{i}",
                axis=-1
            )
            new_nodes.append(softmax)

            # PV MatMul
            context = f"{pattern_name}_context_head_{i}"
            pv_matmul = helper.make_node(
                "MatMul",
                inputs=[probs, v_tensor],
                outputs=[context],
                name=f"{pattern_name}_PV_MatMul_head_{i}"
            )
            new_nodes.append(pv_matmul)
            final_outputs.append(context)

        return new_nodes, final_outputs

    def _concat_heads(
        self, mha_info: Dict, final_outputs: List[str]
    ) -> Tuple[onnx.NodeProto, str]:
        """Create Concat node to merge head outputs."""
        pattern_name = mha_info["pattern"]["QK_MatMul"].replace("_QK_MatMul", "")
        output_name = f"{pattern_name}_context_merged"

        concat_node = helper.make_node(
            "Concat",
            inputs=final_outputs,
            outputs=[output_name],
            name=f"{pattern_name}_Concat_heads",
            axis=0 if self.is_batch_head_swapped else 1
        )

        return concat_node, output_name

    def _rewire_and_cleanup(
        self,
        mha_info: Dict,
        new_nodes: List,
        final_output: str
    ):
        """Rewire graph and remove old nodes."""
        pattern = mha_info['pattern']

        # Nodes to remove
        nodes_to_remove = {pattern["PV_MatMul"], pattern["QK_MatMul"]}
        if pattern["Scale_Q_Mul"]:
            nodes_to_remove.add(pattern["Scale_Q_Mul"])
        if pattern["Scale_K_Mul"]:
            nodes_to_remove.add(pattern["Scale_K_Mul"])
        if pattern["Softmax"]:
            nodes_to_remove.add(pattern["Softmax"])

        # Find original output
        pv_matmul = next(
            (n for n in self.model.graph.node if n.name == pattern["PV_MatMul"]),
            None
        )
        original_output = pv_matmul.output[0] if pv_matmul else None

        # Rewire consumers
        if original_output:
            consumers = self.consumer_map.get(original_output, [])
            for consumer in consumers:
                for i, inp in enumerate(consumer.input):
                    if inp == original_output:
                        consumer.input[i] = final_output

        # Update graph outputs if needed
        for i, out in enumerate(self.model.graph.output):
            if out.name == original_output:
                self.model.graph.output[i].name = final_output

        # Build new node list
        final_nodes = []
        dependency_nodes = {
            pattern["Transpose_Q"], pattern["Transpose_K"], pattern["Transpose_V"]
        }
        seen_deps = set()
        inserted = False

        for node in self.model.graph.node:
            if node.name in nodes_to_remove:
                continue

            final_nodes.append(node)

            # Track dependencies
            if node.name in dependency_nodes:
                seen_deps.add(node.name)

            # Insert new nodes after all dependencies are seen
            if not inserted and len(seen_deps) == 3:
                final_nodes.extend(new_nodes)
                inserted = True

        self.model.graph.ClearField('node')
        self.model.graph.node.extend(final_nodes)
