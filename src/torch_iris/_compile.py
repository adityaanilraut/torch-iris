"""Dynamo backend for Iris graphs using the eager dispatcher and selected fusions."""

import operator

import torch
from torch.fx import Node


def _backward(ctx, grad):
    return grad * ctx.scale, None, None


def _setup_context(ctx, inputs, output):
    ctx.scale = inputs[1]


def _fake(input, scale, bias):
    return torch.empty_like(input)


def _scalar(value):
    return type(value) in (int, float) and -2**24 <= value <= 2**24


def compile_graph(module, example_inputs):
    graph = module.graph
    for add in list(graph.nodes):
        if add.op != "call_function" or add.target is not operator.add or add.kwargs:
            continue
        mul, bias = add.args
        if not isinstance(mul, Node) or not _scalar(bias):
            continue
        if mul.op != "call_function" or mul.target is not operator.mul or mul.kwargs:
            continue
        if mul.next is not add or len(mul.users) != 1:
            continue
        source, scale = mul.args
        if not isinstance(source, Node) or not _scalar(scale):
            continue
        example = source.meta.get("example_value")
        if not isinstance(example, torch.Tensor) or example.device.type != "iris" or example.dtype != torch.float32:
            continue
        with graph.inserting_before(add):
            merged = graph.call_function(torch.ops.torch_iris.scale_add.default, (source, float(scale), float(bias)))
        add.replace_all_uses_with(merged)
        graph.erase_node(add)
        graph.erase_node(mul)
    graph.lint()
    module.recompile()
    return module.forward


def register():
    torch.library.register_fake("torch_iris::scale_add", _fake)
    torch.library.register_autograd("torch_iris::scale_add", _backward, setup_context=_setup_context)
    torch._dynamo.register_backend(compile_graph, name="iris")
