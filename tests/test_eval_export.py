"""Helpers of the export eval: the linear-layer groups of the fp32 graph and bundle paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import onnx
import pytest

from anvil.students.encoder import EncoderClassifier
from eval.export import bundle_dir, linear_groups
from eval.student import TrainedStudent
from tests.test_export import Exported, exported


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Exported:
    return exported(tmp_path_factory)


def test_linear_groups_of_the_tiny_fp32_model(bundle: Exported) -> None:
    model = onnx.load(str(bundle.bundle / "model.onnx"))
    config = EncoderClassifier.load(bundle.student_dir).model.config
    assert (config.hidden_size, config.intermediate_size) == (64, 128)
    groups = linear_groups(model, config)
    for name in ("attention Wqkv", "attention Wo", "MLP Wi", "MLP Wo"):
        assert len(groups[name]) == 2, name  # one per layer
    assert len(groups["head and classifier"]) == 2
    assert len(groups["embeddings"]) == 1
    assert "attention scores" not in groups
    seen = [n for names in groups.values() for n in names]
    assert len(seen) == len(set(seen))
    ops = {n.name: n.op_type for n in model.graph.node}
    assert all(ops[n] in ("MatMul", "Gemm", "Gather") for n in seen)
    weighted = {i.name for i in model.graph.initializer}
    assert len(seen) == sum(
        1
        for n in model.graph.node
        if n.op_type == "Gemm"
        or (n.op_type == "MatMul" and any(x in weighted for x in n.input))
        or (n.op_type == "Gather" and n.input[0] in weighted)
    )


def test_bundle_dir_is_under_the_workspace_exports(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    ts = cast(
        TrainedStudent,
        SimpleNamespace(workspace=SimpleNamespace(root=root), path=root / "students" / "run-7"),
    )
    assert bundle_dir(ts) == root / "exports" / "run-7"


def _parity(variant: str, reference: str, passed: bool) -> dict[str, object]:
    return {
        "variant": variant,
        "reference": reference,
        "n": 10,
        "max_abs_diff": 2e-6,
        "mean_abs_diff": 3e-7,
        "agreement": 1.0,
        "kappa": 0.8,
        "kappa_drop": 0.004,
        "size_mb": 12.0,
        "passed": passed,
        "gate": "a rule",
    }


def test_render_export() -> None:
    from eval.render import render_export

    assert "make eval-export" in render_export([])
    report = {
        "source": {"dataset": "toy", "student_model": "org/tiny", "student_run": "r1"},
        "probes": 10,
        "export_seconds": 3.0,
        "fp32": _parity("fp32", "pytorch", True),
        "int8": _parity("int8", "onnx-fp32", False),
        "int8_fp32_nodes": 2,
        "quantization_probe": {
            "groups": {"MLP Wo": 2},
            "fp32_kappa": 0.8,
            "answered": 7,
            "variants": [
                {
                    "variant": "only MLP Wo",
                    "quantized_nodes": 2,
                    "size_mb": 10.0,
                    "mean_abs_diff": 0.1,
                    "max_abs_diff": 0.5,
                    "agreement": 0.97,
                    "agreement_answered": 1.0,
                    "kappa_drop": 0.02,
                }
            ],
        },
    }
    text = render_export([("export-toy-gold", report, {"command": "c", "git_sha": "abc1234"})])
    assert "| toy | tiny | fp32 | pytorch | 10 | 2.0e-06 | 3.0e-07 | 1.000 | n/a |" in text
    assert "| int8 |" in text
    assert "failed |" in text
    # Reports without a per-variant count fall back to the fp32 count in the probe's head.
    assert "| only MLP Wo | 2 | 10 MB | 0.100 | 0.500 | 0.970 | 1.000 (n=7) | +0.020 |" in text
    assert "| Agreement on answered inputs |" in text


def test_quantization_probe_scores_every_group(bundle: Exported) -> None:
    from eval.export import quantization_probe
    from tests.test_export import PROBES, TEACHER

    ts = cast(TrainedStudent, SimpleNamespace(path=bundle.student_dir))
    head, *rows = quantization_probe(ts, bundle.bundle, list(PROBES), list(TEACHER))
    groups = cast(dict[str, int], head["groups"])
    names = [r["variant"] for r in rows]
    assert names[:3] == [
        "everything",
        "default: all but MLP Wo",
        "all but MLP Wo and embeddings",
    ]
    assert names[3:] == [f"only {g}" for g in groups]
    total = sum(groups.values())
    assert rows[0]["quantized_nodes"] == total
    assert rows[1]["fp32_nodes"] == groups["MLP Wo"]
    for r in rows:
        assert 0.0 <= cast(float, r["agreement"]) <= 1.0
        # Every variant counts at least the inputs fp32 answers at the bundle's threshold.
        assert cast(int, head["answered"]) <= cast(int, r["answered"]) <= len(PROBES)
        assert cast(int, r["quantized_nodes"]) + cast(int, r["fp32_nodes"]) == total


def test_linear_groups_tell_wqkv_from_mlp_wi_when_their_shapes_match() -> None:
    # ModernBERT-base has hidden 768 and intermediate 1152: Wqkv and MLP Wi are both
    # [768, 2304]. Here hidden 4 and intermediate 6 give the same collision on two layers.
    from onnx import TensorProto, helper

    def init(name: str, dims: list[int]) -> onnx.TensorProto:
        return helper.make_tensor(name, TensorProto.FLOAT, dims, [0.0] * (dims[0] * dims[1]))

    nodes, inits = [], []
    for layer in range(2):
        for role, dims in (("qkv", [4, 12]), ("wo", [4, 4]), ("wi", [4, 12]), ("down", [6, 4])):
            w = f"w_{role}_{layer}"
            inits.append(init(w, dims))
            nodes.append(helper.make_node("MatMul", ["x", w], [f"y_{role}_{layer}"], name=w))
    graph = helper.make_graph(nodes, "g", [], [], initializer=inits)
    config = SimpleNamespace(hidden_size=4, intermediate_size=6)
    groups = linear_groups(helper.make_model(graph), config)
    assert groups["attention Wqkv"] == ["w_qkv_0", "w_qkv_1"]
    assert groups["MLP Wi"] == ["w_wi_0", "w_wi_1"]
    assert groups["attention Wo"] == ["w_wo_0", "w_wo_1"]
    assert groups["MLP Wo"] == ["w_down_0", "w_down_1"]
