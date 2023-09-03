from typing import Dict, Generator, Optional, Set, Tuple

import networkx as nx

import numpy as np

from collector import Collector


class Leafer:
    def __init__(self, G: nx.DiGraph) -> None:
        self.G = G

    def find_parents(
        self,
        start_node: str,
        depth: int,
        level: int = 0,
        parents: Optional[Dict[int, Set[str]]] = None,
        unique_parents: Optional[Set[str]] = None,
    ) -> Dict[int, Set[str]]:
        if parents is None:
            parents = {}
            unique_parents = set()

        for node in self.G.predecessors(start_node):
            if node not in unique_parents:
                # print(node, level)

                unique_parents.add(node)

                if level in parents.keys():
                    parents[level].add(node)
                else:
                    parents[level] = set([node])

                if level + 1 < depth:
                    self.find_parents(node, depth, level + 1, parents, unique_parents)

        return parents

    def split_train_test(
        self,
        generation_depth: int = 40,
        ancestors_depth: int = 2,
        p=0.1,
        p_divide_leafs=0.5,
        min_to_test_rate=0.5,
        weights=[0.2, 0.2, 0.2, 0.2, 0.2],
    ):
        """
        Interface for train test splitting
        """
        self.collector = Collector(
            self.G,
            generation_depth,
            ancestors_depth,
            p,
            p_divide_leafs,
            min_to_test_rate,
            weights=weights,
        )
        # self.collector.collect_only_child()
        # self.collector.collect_only_leafs()
        # self.collector.collect_not_only_leafs()
        # # if triplets needed
        # self.collector.collect_simple_triplets()
        # self.collector.collect_mixes_triplets()
        self.collector.collect_possible_samples()

        return self.collector.train, self.collector.test
