from spins.goos import graph_executor


def test_top_sort_affinity():
    graph = {
        "a": ["b", "c", "d"],
        "b": ["h"],
        "c": ["g"],
        "d": ["e"],
        "e": ["f"],
        "f": ["g"],
        "g": ["h"],
        "h": []
    }
    affinity_nodes = set(["b", "g", "f"])

    sorted_nodes = graph_executor._top_sort_affinity(graph, affinity_nodes)

    # The topological ordering must be a[cde][bf]gh where square brackets denote
    # that any combination is acceptable.
    assert sorted_nodes[0] == "a"
    assert set(sorted_nodes[1:4]) == set(["c", "d", "e"])
    assert set(sorted_nodes[4:6]) == set(["b", "f"])
    assert sorted_nodes[6] == "g"
    assert sorted_nodes[7] == "h"
