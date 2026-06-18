from __future__ import annotations

import onnx
from onnx import TensorProto, helper

from ..base import Logger, OptimizationPass


class FeatureConcatSliceCleanupPass(OptimizationPass):
    """Remove redundant DA3 intermediate-feature concat/slice pairs.

    DA3's backbone emits cat-token features as concat(local, global). The export
    graph then slices that concat immediately to recover the two halves before
    concatenating them again for the DPT head:

      Concat_6(A, B) -> Slice_1(:dim), Slice_2(dim:)

    The two slices are pure views of the concat inputs, so they can be replaced
    directly by A and B. This avoids an unnecessary large concat and two slices
    per intermediate feature stage.
    """

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header("Running DA3 Feature Concat/Slice Cleanup Pass")
        self.model = model
        self._build_maps()

        single_view_processed = self._run_single_view_concat_slice_cleanup()
        if single_view_processed:
            self._build_maps()
        multi_view_processed = self._run_multi_view_concat_slice_cleanup()

        Logger.success(
            "Feature concat/slice cleanup complete: "
            f"{single_view_processed} single-view stage(s), "
            f"{multi_view_processed} multi-view stage(s)"
        )
        return self.model

    def _run_single_view_concat_slice_cleanup(self) -> int:
        replacements: dict[str, str] = {}
        remove_names: set[str] = set()
        processed = 0

        for concat in list(self.model.graph.node):
            if concat.op_type != "Concat" or not concat.name.startswith("/model/backbone/Concat_"):
                continue
            if len(concat.input) != 2 or not concat.output:
                continue
            consumers = self.consumer_map.get(concat.output[0], [])
            if len(consumers) != 2 or any(node.op_type != "Slice" for node in consumers):
                continue

            split_dim = self._last_dim(concat.input[0])
            if split_dim is None:
                continue

            lower = None
            upper = None
            for node in consumers:
                start = self._const_scalar(node.input[1]) if len(node.input) > 1 else None
                end = self._const_scalar(node.input[2]) if len(node.input) > 2 else None
                axis = self._const_scalar(node.input[3]) if len(node.input) > 3 else None
                if axis not in {-1, 3}:
                    continue
                if start == 0 and end == split_dim:
                    lower = node
                elif start == split_dim:
                    upper = node

            if not lower or not upper:
                continue

            replacements[lower.output[0]] = concat.input[0]
            replacements[upper.output[0]] = concat.input[1]
            remove_names.update({concat.name, lower.name, upper.name})
            processed += 1

        if replacements:
            for node in self.model.graph.node:
                if node.name in remove_names:
                    continue
                for idx, input_name in enumerate(node.input):
                    if input_name in replacements:
                        node.input[idx] = replacements[input_name]
            keep = [node for node in self.model.graph.node if node.name not in remove_names]
            self.model.graph.ClearField("node")
            self.model.graph.node.extend(keep)

        return processed

    def _run_multi_view_concat_slice_cleanup(self) -> int:
        remove_names: set[str] = set()
        insertion_after: dict[str, list[onnx.NodeProto]] = {}
        processed = 0

        for final_slice in list(self.model.graph.node):
            pattern = self._match_multi_view_feature_slice(final_slice)
            if not pattern:
                continue

            final_out = final_slice.output[0]
            local_patch = f"{final_out}_mv_local_patch"
            global_patch = f"{final_out}_mv_global_patch"
            starts = self._const_int64(f"{final_slice.name}_mv_patch_starts", [1])
            ends = self._const_int64(f"{final_slice.name}_mv_patch_ends", [9223372036854775807])
            axes = self._const_int64(f"{final_slice.name}_mv_patch_axes", [2])
            steps = self._const_int64(f"{final_slice.name}_mv_patch_steps", [1])

            global_norm = pattern["global_norm"]
            global_norm.input[0] = pattern["global_rank4"]

            insertion_after.setdefault(global_norm.name, []).extend(
                [
                    helper.make_node(
                        "Slice",
                        [pattern["local_rank4"], starts, ends, axes, steps],
                        [local_patch],
                        name=f"{final_slice.name}_mv_local_patch",
                    ),
                    helper.make_node(
                        "Slice",
                        [global_norm.output[0], starts, ends, axes, steps],
                        [global_patch],
                        name=f"{final_slice.name}_mv_global_patch",
                    ),
                    helper.make_node(
                        "Concat",
                        [local_patch, global_patch],
                        [final_out],
                        name=f"{final_slice.name}_mv_feature_concat",
                        axis=-1,
                    ),
                ]
            )

            remove_names.update(
                {
                    final_slice.name,
                    pattern["final_concat"].name,
                    pattern["local_slice"].name,
                    pattern["global_slice"].name,
                    pattern["source_reshape"].name,
                    pattern["gather"].name,
                    pattern["flatten"].name,
                    pattern["source_concat"].name,
                }
            )
            processed += 1

        if insertion_after:
            final_nodes = []
            for node in self.model.graph.node:
                if node.name in remove_names:
                    continue
                final_nodes.append(node)
                if node.name in insertion_after:
                    final_nodes.extend(insertion_after[node.name])
            self.model.graph.ClearField("node")
            self.model.graph.node.extend(final_nodes)

        return processed

    def _match_multi_view_feature_slice(self, final_slice: onnx.NodeProto) -> dict | None:
        if final_slice.op_type != "Slice" or not final_slice.name.startswith("/model/backbone/Slice_"):
            return None
        if len(final_slice.input) < 5 or not final_slice.output:
            return None
        if self._const_scalar(final_slice.input[1]) != 1:
            return None
        if self._const_scalar(final_slice.input[3]) != 2:
            return None
        final_shape = self._shape(final_slice.output[0])
        if final_shape is None or len(final_shape) != 4 or final_shape[1] <= 1:
            return None

        final_concat = self.producer_map.get(final_slice.input[0])
        if (
            not final_concat
            or final_concat.op_type != "Concat"
            or len(final_concat.input) != 2
            or self._axis(final_concat) not in {-1, 3}
        ):
            return None

        local_slice = self.producer_map.get(final_concat.input[0])
        global_norm = self.producer_map.get(final_concat.input[1])
        if not local_slice or local_slice.op_type != "Slice":
            return None
        if not global_norm or global_norm.op_type != "LayerNormalization":
            return None
        global_slice = self.producer_map.get(global_norm.input[0])
        if not global_slice or global_slice.op_type != "Slice":
            return None
        if local_slice.input[0] != global_slice.input[0]:
            return None

        source_reshape = self.producer_map.get(local_slice.input[0])
        if not source_reshape or source_reshape.op_type != "Reshape":
            return None
        gather = self.producer_map.get(source_reshape.input[0])
        if not gather or gather.op_type != "Gather" or self._axis(gather) != 0:
            return None
        flatten = self.producer_map.get(gather.input[0])
        if not flatten or flatten.op_type != "Flatten" or self._axis(flatten) != 2:
            return None
        source_concat = self.producer_map.get(flatten.input[0])
        if (
            not source_concat
            or source_concat.op_type != "Concat"
            or len(source_concat.input) != 2
            or self._axis(source_concat) not in {-1, 3}
        ):
            return None

        source_shape = self._shape(source_concat.output[0])
        reshape_shape = self._shape(source_reshape.output[0])
        if not source_shape or source_shape != reshape_shape or len(source_shape) != 4:
            return None
        if source_shape[0] != 1 or source_shape[1] <= 1:
            return None
        if not self._is_identity_view_gather(gather.input[1], source_shape[1]):
            return None

        split_dim = self._last_dim(source_concat.input[0])
        if split_dim is None:
            return None
        if not self._is_lower_slice(local_slice, split_dim):
            return None
        if not self._is_upper_slice(global_slice, split_dim):
            return None

        if any(node.name != final_slice.name for node in self.consumer_map.get(final_concat.output[0], [])):
            return None
        if any(node.name != final_concat.name for node in self.consumer_map.get(local_slice.output[0], [])):
            return None
        if any(node.name != global_norm.name for node in self.consumer_map.get(global_slice.output[0], [])):
            return None
        if any(node.name != final_concat.name for node in self.consumer_map.get(global_norm.output[0], [])):
            return None
        if any(node.name != source_reshape.name for node in self.consumer_map.get(gather.output[0], [])):
            return None
        if any(node.name != gather.name for node in self.consumer_map.get(flatten.output[0], [])):
            return None
        if any(node.name != flatten.name for node in self.consumer_map.get(source_concat.output[0], [])):
            return None

        local_rank4, global_rank4 = source_concat.input
        local_shape = self._shape(local_rank4)
        global_shape = self._shape(global_rank4)
        if local_shape is None or global_shape is None or local_shape != global_shape:
            return None
        if len(local_shape) != 4 or local_shape[-1] != split_dim:
            return None

        return {
            "final_concat": final_concat,
            "local_slice": local_slice,
            "global_slice": global_slice,
            "global_norm": global_norm,
            "source_reshape": source_reshape,
            "gather": gather,
            "flatten": flatten,
            "source_concat": source_concat,
            "local_rank4": local_rank4,
            "global_rank4": global_rank4,
        }

    def _axis(self, node: onnx.NodeProto) -> int | None:
        for attr in node.attribute:
            if attr.name == "axis":
                return int(attr.i)
        return None

    def _is_identity_view_gather(self, tensor_name: str, views: int) -> bool:
        init = next((init for init in self.model.graph.initializer if init.name == tensor_name), None)
        if init is None:
            return False
        from onnx import numpy_helper

        arr = numpy_helper.to_array(init).reshape(-1)
        return arr.size == views and arr.tolist() == list(range(views))

    def _is_lower_slice(self, node: onnx.NodeProto, split_dim: int) -> bool:
        axis = self._const_scalar(node.input[3]) if len(node.input) > 3 else None
        start = self._const_scalar(node.input[1]) if len(node.input) > 1 else None
        end = self._const_scalar(node.input[2]) if len(node.input) > 2 else None
        return axis in {-1, 3} and start == 0 and end == split_dim

    def _is_upper_slice(self, node: onnx.NodeProto, split_dim: int) -> bool:
        axis = self._const_scalar(node.input[3]) if len(node.input) > 3 else None
        start = self._const_scalar(node.input[1]) if len(node.input) > 1 else None
        return axis in {-1, 3} and start == split_dim

    def _shape(self, tensor_name: str) -> list[int] | None:
        for value_info in (
            list(self.model.graph.value_info)
            + list(self.model.graph.input)
            + list(self.model.graph.output)
        ):
            if value_info.name != tensor_name:
                continue
            dims = value_info.type.tensor_type.shape.dim
            if not dims:
                return None
            out: list[int] = []
            for dim in dims:
                if not dim.HasField("dim_value"):
                    return None
                out.append(int(dim.dim_value))
            return out
        init = next((init for init in self.model.graph.initializer if init.name == tensor_name), None)
        if init is not None:
            return list(init.dims)
        return None

    def _const_int64(self, name: str, values: list[int]) -> str:
        if name not in {init.name for init in self.model.graph.initializer}:
            self.model.graph.initializer.extend(
                [helper.make_tensor(name, TensorProto.INT64, [len(values)], values)]
            )
        return name

    def _last_dim(self, tensor_name: str) -> int | None:
        shape = self._shape(tensor_name)
        return shape[-1] if shape else None

    def _const_scalar(self, tensor_name: str) -> int | None:
        init = next((init for init in self.model.graph.initializer if init.name == tensor_name), None)
        if init is not None:
            from onnx import numpy_helper

            arr = numpy_helper.to_array(init)
            return int(arr.reshape(-1)[0]) if arr.size else None
        producer = self.producer_map.get(tensor_name)
        if producer and producer.op_type == "Constant" and producer.attribute:
            from onnx import numpy_helper

            arr = numpy_helper.to_array(producer.attribute[0].t)
            return int(arr.reshape(-1)[0]) if arr.size else None
        return None


class FeatureRank3CleanupPass(OptimizationPass):
    """Keep DA3 single-view intermediate features in rank-3 form.

    DA3's official backbone keeps an explicit view dimension and exports DPT
    features as [B, S, N, C]. This repo exports DA3 with S=1, so several feature
    taps round-trip through [1, 1, 1370, C] only to return to [1, 1370, C].
    This pass rewrites each DPT feature tap to:

      global_tokens = LayerNorm(global_tokens)
      feature_all   = Concat(local_tokens, global_tokens, axis=2)
      feature       = Slice(feature_all, axis=1, start=1)

    This preserves the class-token crop and local/global channel concat while
    avoiding rank-4 concat/slice/reshape traffic in the head input path.
    """

    def run(self, model: onnx.ModelProto) -> onnx.ModelProto:
        Logger.header("Running DA3 Feature Rank-3 Cleanup Pass")
        self.model = model
        self._build_maps()

        replacements: dict[str, str] = {}
        remove_names: set[str] = set()
        insertion_after: dict[str, list[onnx.NodeProto]] = {}
        processed = 0

        for concat in list(self.model.graph.node):
            pattern = self._match_feature_concat(concat)
            if not pattern:
                continue

            merged_all = f"{concat.output[0]}_rank3_all_tokens"
            merged = f"{concat.output[0]}_rank3"
            global_norm_out = f"{pattern['global_norm'].output[0]}_rank3"
            starts = self._const_int64(f"{concat.name}_rank3_slice_starts", [1])
            ends = self._const_int64(f"{concat.name}_rank3_slice_ends", [9223372036854775807])
            axes = self._const_int64(f"{concat.name}_rank3_slice_axes", [1])
            steps = self._const_int64(f"{concat.name}_rank3_slice_steps", [1])

            pattern["global_norm"].input[0] = pattern["global_tokens"]
            pattern["global_norm"].output[0] = global_norm_out
            insertion_after.setdefault(pattern["global_norm"].name, []).extend(
                [
                    helper.make_node(
                        "Concat",
                        [pattern["local_tokens"], global_norm_out],
                        [merged_all],
                        name=f"{concat.name}_rank3",
                        axis=2,
                    ),
                    helper.make_node(
                        "Slice",
                        [merged_all, starts, ends, axes, steps],
                        [merged],
                        name=f"{concat.name}_patch_rank3",
                    ),
                ]
            )

            replacements[pattern["head_reshape"].output[0]] = merged
            if pattern["local_rank3_reshape"] is not None:
                replacements[pattern["local_rank3_reshape"].output[0]] = pattern["local_tokens"]
            if pattern["global_rank3_reshape"] is not None:
                replacements[pattern["global_rank3_reshape"].output[0]] = pattern["global_tokens"]

            remove_names.update(
                {
                    concat.name,
                    pattern["slice"].name,
                    pattern["head_reshape"].name,
                    pattern["local_rank4_reshape"].name,
                    pattern["global_rank4_reshape"].name,
                }
            )
            remove_names.update(pattern["extra_remove"])
            if pattern["local_rank3_reshape"] is not None:
                remove_names.add(pattern["local_rank3_reshape"].name)
            if pattern["global_rank3_reshape"] is not None:
                remove_names.add(pattern["global_rank3_reshape"].name)
            processed += 1

        if replacements or insertion_after:
            for node in self.model.graph.node:
                if node.name in remove_names:
                    continue
                for idx, input_name in enumerate(node.input):
                    if input_name in replacements:
                        node.input[idx] = replacements[input_name]

            final_nodes = []
            for node in self.model.graph.node:
                if node.name in remove_names:
                    continue
                final_nodes.append(node)
                if node.name in insertion_after:
                    final_nodes.extend(insertion_after[node.name])

            self.model.graph.ClearField("node")
            self.model.graph.node.extend(final_nodes)

        Logger.success(f"Feature rank-3 cleanup complete: {processed} stage(s)")
        return self.model

    def _match_feature_concat(self, concat: onnx.NodeProto) -> dict | None:
        if concat.op_type != "Concat" or not concat.name.startswith("/model/backbone/Concat_"):
            return None
        axis = next((int(attr.i) for attr in concat.attribute if attr.name == "axis"), None)
        if axis not in {-1, 3} or len(concat.input) != 2 or not concat.output:
            return None

        consumers = self.consumer_map.get(concat.output[0], [])
        if len(consumers) != 1 or consumers[0].op_type != "Slice":
            return None
        slice_node = consumers[0]
        if self._const_scalar(slice_node.input[1]) != 1:
            return None
        if self._const_scalar(slice_node.input[3]) != 2:
            return None

        head_consumers = self.consumer_map.get(slice_node.output[0], [])
        if len(head_consumers) != 1 or head_consumers[0].op_type != "Reshape":
            return None
        head_reshape = head_consumers[0]

        global_norm = self.producer_map.get(concat.input[1])
        if not global_norm or global_norm.op_type != "LayerNormalization":
            return None
        pair = self._resolve_split_pair(concat.input[0], global_norm.input[0], current_concat=concat.name)
        if pair:
            local = pair["local"]
            global_rank4 = pair["global"]["rank4_reshape"]
            global_tokens = pair["global"]["tokens"]
            extra_remove = set(pair["extra_remove"])
        else:
            local = self._resolve_local_rank4_feature(concat.input[0], current_concat=concat.name)
            if not local:
                return None
            global_rank4 = self.producer_map.get(global_norm.input[0])
            if not global_rank4 or global_rank4.op_type != "Reshape":
                return None
            global_tokens = global_rank4.input[0]
            extra_remove = set(local["extra_remove"])
        if any(node.name != concat.name for node in self.consumer_map.get(global_norm.output[0], [])):
            return None

        local_rank4 = local["rank4_reshape"]
        local_tokens = local["tokens"]
        if self._rank(local_tokens) != 3 or self._rank(global_tokens) != 3:
            return None
        if self._shape(local_rank4.output[0]) != [1, 1, 1370, self._shape(local_tokens)[-1]]:
            return None
        if self._shape(global_rank4.output[0]) != [1, 1, 1370, self._shape(global_tokens)[-1]]:
            return None

        return {
            "slice": slice_node,
            "head_reshape": head_reshape,
            "local_rank4_reshape": local_rank4,
            "global_rank4_reshape": global_rank4,
            "local_rank3_reshape": self._rank3_consumer(local_rank4.output[0], exclude=local["exclude_rank3"]),
            "global_rank3_reshape": self._rank3_consumer(global_rank4.output[0], exclude=global_norm.name),
            "global_norm": global_norm,
            "local_tokens": local_tokens,
            "global_tokens": global_tokens,
            "extra_remove": extra_remove,
        }

    def _resolve_split_pair(self, local_tensor: str, global_tensor: str, current_concat: str) -> dict | None:
        local_slice = self.producer_map.get(local_tensor)
        global_slice = self.producer_map.get(global_tensor)
        if not local_slice or not global_slice:
            return None
        if local_slice.op_type != "Slice" or global_slice.op_type != "Slice":
            return None
        if local_slice.input[0] != global_slice.input[0]:
            return None
        source_concat = self.producer_map.get(local_slice.input[0])
        if not source_concat or source_concat.op_type != "Concat" or len(source_concat.input) != 2:
            return None
        first = self.producer_map.get(source_concat.input[0])
        second = self.producer_map.get(source_concat.input[1])
        if not first or not second or first.op_type != "Reshape" or second.op_type != "Reshape":
            return None
        split_dim = self._last_dim(source_concat.input[0])
        if split_dim is None:
            return None
        if not self._is_lower_slice(local_slice, split_dim):
            return None
        if not self._is_upper_slice(global_slice, split_dim):
            return None
        for node in self.consumer_map.get(source_concat.output[0], []):
            if node.name not in {local_slice.name, global_slice.name}:
                return None
        for consumer in self.consumer_map.get(local_slice.output[0], []):
            if consumer.name != current_concat:
                return None
        for consumer in self.consumer_map.get(global_slice.output[0], []):
            if consumer.op_type != "LayerNormalization":
                return None
        return {
            "local": {
                "rank4_reshape": first,
                "tokens": first.input[0],
                "exclude_rank3": source_concat.name,
                "extra_remove": set(),
            },
            "global": {
                "rank4_reshape": second,
                "tokens": second.input[0],
            },
            "extra_remove": {source_concat.name, local_slice.name, global_slice.name},
        }

    def _resolve_local_rank4_feature(self, tensor: str, current_concat: str) -> dict | None:
        producer = self.producer_map.get(tensor)
        if producer is None:
            return None
        if producer.op_type == "Reshape":
            return {
                "rank4_reshape": producer,
                "tokens": producer.input[0],
                "exclude_rank3": current_concat,
                "extra_remove": set(),
            }
        if producer.op_type != "Slice":
            return None

        source_concat = self.producer_map.get(producer.input[0])
        if not source_concat or source_concat.op_type != "Concat" or len(source_concat.input) != 2:
            return None
        axis = self._const_scalar(producer.input[3]) if len(producer.input) > 3 else None
        if axis not in {-1, 3}:
            return None
        first_rank4 = self.producer_map.get(source_concat.input[0])
        if not first_rank4 or first_rank4.op_type != "Reshape":
            return None
        split_dim = self._last_dim(source_concat.input[0])
        if split_dim is None:
            return None
        start = self._const_scalar(producer.input[1]) if len(producer.input) > 1 else None
        end = self._const_scalar(producer.input[2]) if len(producer.input) > 2 else None
        if start != 0 or end != split_dim:
            return None

        source_consumers = self.consumer_map.get(source_concat.output[0], [])
        if not source_consumers or any(node.op_type != "Slice" for node in source_consumers):
            return None
        for node in source_consumers:
            consumers = self.consumer_map.get(node.output[0], [])
            if node.name == producer.name:
                if any(consumer.name != current_concat for consumer in consumers):
                    return None
            elif consumers:
                return None

        return {
            "rank4_reshape": first_rank4,
            "tokens": first_rank4.input[0],
            "exclude_rank3": source_concat.name,
            "extra_remove": {source_concat.name, *(node.name for node in source_consumers)},
        }

    def _is_lower_slice(self, node: onnx.NodeProto, split_dim: int) -> bool:
        axis = self._const_scalar(node.input[3]) if len(node.input) > 3 else None
        start = self._const_scalar(node.input[1]) if len(node.input) > 1 else None
        end = self._const_scalar(node.input[2]) if len(node.input) > 2 else None
        return axis in {-1, 3} and start == 0 and end == split_dim

    def _is_upper_slice(self, node: onnx.NodeProto, split_dim: int) -> bool:
        axis = self._const_scalar(node.input[3]) if len(node.input) > 3 else None
        start = self._const_scalar(node.input[1]) if len(node.input) > 1 else None
        return axis in {-1, 3} and start == split_dim

    def _rank3_consumer(self, tensor: str, exclude: str | set[str]) -> onnx.NodeProto | None:
        excludes = {exclude} if isinstance(exclude, str) else exclude
        for node in self.consumer_map.get(tensor, []):
            if node.name in excludes:
                continue
            if node.op_type == "Reshape" and self._rank(node.output[0]) == 3:
                return node
        return None

    def _shape(self, tensor_name: str) -> list[int] | None:
        for value_info in (
            list(self.model.graph.value_info)
            + list(self.model.graph.input)
            + list(self.model.graph.output)
        ):
            if value_info.name != tensor_name:
                continue
            dims = value_info.type.tensor_type.shape.dim
            if not dims:
                return None
            out: list[int] = []
            for dim in dims:
                if not dim.HasField("dim_value"):
                    return None
                out.append(int(dim.dim_value))
            return out
        return None

    def _rank(self, tensor_name: str) -> int | None:
        shape = self._shape(tensor_name)
        return len(shape) if shape is not None else None

    def _const_int64(self, name: str, values: list[int]) -> str:
        if name not in {init.name for init in self.model.graph.initializer}:
            self.model.graph.initializer.extend(
                [helper.make_tensor(name, TensorProto.INT64, [len(values)], values)]
            )
        return name

    def _const_scalar(self, tensor_name: str) -> int | None:
        init = next((init for init in self.model.graph.initializer if init.name == tensor_name), None)
        if init is not None:
            from onnx import numpy_helper

            arr = numpy_helper.to_array(init)
            return int(arr.reshape(-1)[0]) if arr.size else None
        producer = self.producer_map.get(tensor_name)
        if producer and producer.op_type == "Constant" and producer.attribute:
            from onnx import numpy_helper

            arr = numpy_helper.to_array(producer.attribute[0].t)
            return int(arr.reshape(-1)[0]) if arr.size else None
        return None

    def _last_dim(self, tensor_name: str) -> int | None:
        shape = self._shape(tensor_name)
        return shape[-1] if shape else None
