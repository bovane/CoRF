"""
partition_utils.py

Unified partition dispatcher.

Supported:
    - corf
    - corf_component (experimental component-aware BLOCS)
    - random
    - metis
    - cluster_gcn
    - graphsaint_rw
    - louvain

CoRF keeps its original core logic.
Other methods optionally apply shared-core augmentation.
"""

from ultra.partition_corf import build_partition_corf
from ultra.partition_random import build_partition_random
from ultra.partition_metis import build_partition_metis
from ultra.partition_cluster_gcn import build_partition_cluster_gcn
from ultra.partition_graphsaint_rw import build_partition_graphsaint_rw
from ultra.partition_louvain import build_partition_louvain
from ultra.partition_core import apply_shared_core_if_enabled


def _normalize_partition_method(method):

    if isinstance(method, int):
        mapping = {
            0: "corf",
            1: "random",
            2: "metis",
            3: "cluster_gcn",
            4: "graphsaint_rw",
            5: "louvain",
        }

        if method not in mapping:
            raise ValueError(f"Unknown partition id: {method}")

        method = mapping[method]

    method = str(method).strip().lower()

    aliases = {
        "corf": "corf",
        "random": "random",
        "metis": "metis",
        "cluster_gcn": "cluster_gcn",
        "graphsaint_rw": "graphsaint_rw",
        "louvain": "louvain",
    }

    method = aliases.get(method, method)

    supported = {
        "corf",
        "corf_component",
        "random",
        "metis",
        "cluster_gcn",
        "graphsaint_rw",
        "louvain",
    }

    if method not in supported:
        raise ValueError(f"Unknown partition method: {method}")

    return method


def _apply_non_corf_shared_core(
    pyg_data,
    records,
    method,
    **kwargs,
):

    if method == "corf":
        return records

    if not kwargs["core_merge"]:
        return records

    core_ratio = float(kwargs["core_node_proportions"])

    if not 0.0 <= core_ratio <= 1.0:
        raise ValueError(
            f"core_node_proportions must be in [0, 1], got {core_ratio}"
        )

    if core_ratio == 0.0:
        return records

    print(
        f"[partition-core] method={method} | "
        f"core_ratio={core_ratio} | "
        f"selection={kwargs['core_selection']}"
    )

    return apply_shared_core_if_enabled(
        pyg_data=pyg_data,
        records=records,
        core_merge=kwargs["core_merge"],
        core_node_proportions=core_ratio,
        core_selection=kwargs["core_selection"],
        method_tag=method,
    )


def build_partition(
    method,
    pyg_data,
    **kwargs,
):

    method = _normalize_partition_method(method)
    component_core = kwargs.pop("component_core", False)
    repair = kwargs.pop("repair_core_paths", False)
    capture = kwargs.pop("capture_base_nodes", False) or repair
    budgets = dict(max_hops=kwargs.pop("path_max_hops", 3),
                   max_new_nodes=kwargs.pop("path_max_new_nodes", 128),
                   max_edge_scans=kwargs.pop("path_max_edge_scans", 200000))
    if component_core and method not in {"corf", "corf_component", "cluster_gcn"}:
        raise ValueError("Component core is currently supported for corf and cluster_gcn only")
    if repair and not kwargs["core_merge"]:
        raise ValueError("Path repair requires core_merge=True")
    def finish(records):
        if repair:
            from ultra.core_path_repair import repair_records
            return repair_records(pyg_data, records, **budgets)
        return records

    print(f"[partition] method={method} component_core={component_core} path_repair={repair}")
    if method in {"corf", "corf_component"}:
        return finish(build_partition_corf(
            pyg_data, component_aware=component_core or method == "corf_component",
            capture_base_nodes=capture, **kwargs))

    if method == "random":
        records = build_partition_random(
            pyg_data,
            kwargs["subgraph_max_size"],
            kwargs["seed"],
        )

    elif method == "metis":
        records = build_partition_metis(
            pyg_data,
            kwargs["subgraph_max_size"],
            seed=kwargs["seed"],
        )

    elif method == "cluster_gcn":
        records = build_partition_cluster_gcn(
            pyg_data,
            kwargs["subgraph_max_size"],
        )

    elif method == "graphsaint_rw":
        records = build_partition_graphsaint_rw(
            pyg_data,
            kwargs["subgraph_max_size"],
            kwargs["seed"],
        )

    elif method == "louvain":
        records = build_partition_louvain(
            pyg_data,
            kwargs["subgraph_max_size"],
            seed=kwargs["seed"],
        )

    else:
        raise RuntimeError(
            f"Unhandled partition method: {method}"
        )

    if method == "cluster_gcn" and component_core and kwargs["core_merge"]:
        from ultra.component_core_merge import merge_cluster_component_core
        return finish(merge_cluster_component_core(
            pyg_data, records, kwargs["core_node_proportions"], kwargs["core_selection"]))
    base_parts = [r.nodes.copy() for r in records] if capture else None
    records = _apply_non_corf_shared_core(pyg_data=pyg_data, records=records, method=method, **kwargs)
    if capture:
        import numpy as np
        from ultra.partition_core import select_core_nodes
        core = (select_core_nodes(pyg_data, kwargs["core_node_proportions"], kwargs["core_selection"])
                if kwargs["core_merge"] else np.empty(0, dtype=np.int64))
        for record, base in zip(records, base_parts):
            record.metadata.update(base_nodes=base, core_nodes=core.copy())
    return finish(records)



__all__ = [
    "build_partition",
]