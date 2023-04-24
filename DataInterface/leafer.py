from typing import Dict, Generator, Optional, Set, Tuple

import networkx as nx


class Leafer:
    def __init__(
        self, G: nx.DiGraph, generation_depth: int = 40, ancestors_depth: int = 2
    ) -> None:
        self.generation_depth = generation_depth
        self.ancestors_depth = ancestors_depth
        self.G = G

        self.precompute_generations()

    def precompute_generations(self) -> None:
        self.generations = {}
        for i, gen in enumerate(nx.topological_generations(self.G)):
            for word in gen:
                self.generations[word] = i

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

    def leafs_generator(self) -> Generator[Tuple[str, Dict[int, Set[str]]], None, None]:
        for node, degree in self.G.out_degree():
            if (
                degree == 0
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                parents = self.find_parents(node, self.ancestors_depth)

                yield ((node, parents))
