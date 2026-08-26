"""检查 ONNX 图的输出节点上游结构（定位真实通道布局）。"""

import onnx

model = onnx.load(r"D:\Workspace\SarDetection\t2_qgis_intelligent\yolo11n-obb.onnx")
graph = model.graph

for out in graph.output:
    print("graph output:", out.name, [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim])

# 输出张量由哪个节点产生 → 打印其算子与属性
producer = {node.output[0]: node for node in graph.node}
for out in graph.output:
    node = producer.get(out.name)
    if node is None:
        print(f"{out.name}: (graph input / constant)")
        continue
    print(f"\n{out.name} <- op={node.op_type}")
    if node.op_type in ("Concat", "Slice", "Reshape"):
        print("  inputs:", node.input)
        for inp in node.input:
            up = producer.get(inp)
            if up is not None:
                print(f"    {inp} <- {up.op_type} ({list(up.input)[:3]})")
    for attr in node.attribute:
        if attr.name in ("axis", "starts", "ends", "perm"):
            print("  attr", attr.name, onnx.helper.get_attribute_value(attr))
