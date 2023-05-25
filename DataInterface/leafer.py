from typing import Dict, Generator, Optional, Set, Tuple

import networkx as nx

import numpy as np


class GeneratorMixin:
    def leafs_generator(self) -> Generator[Tuple[str, Dict[int, Set[str]]], None, None]:
        for node, degree in self.G.out_degree():
            if (
                degree == 0
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                parents = self.find_parents(node, self.ancestors_depth)

                yield ((node, parents))

    def only_child_generator(self) -> None:
        """
        Generator function that return verteces that has only one leaf children
        """
        for node, degree in self.G.out_degree():
            if (
                degree == 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                for child in self.G.successors(node):
                    if self.G.out_degree(child) == 0:
                        yield (child, node, list(self.G.predecessors(node)))

    def all_children_leafs_generator(self):
        """
        Generator function that returns vertices that are all children of one parent
        and they are all leafs
        """
        for node, degree in self.G.out_degree():
            if (
                degree > 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                all_children_leafs = True
                for child in self.G.successors(node):
                    if self.G.out_degree(child) > 0:
                        all_children_leafs = False
                        break

                if all_children_leafs:
                    yield (node, list(self.G.successors(node)))

    def leaf_and_no_leaf_generator(self):
        """
        Generator function that returns verteces that are leafs but other children
        of their parents are not leafs
        """
        for node, degree in self.G.out_degree():
            if (
                degree > 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                all_children_leafs = True
                exists_a_leaf = False
                for child in self.G.successors(node):
                    if self.G.out_degree(child) > 0:
                        all_children_leafs = False
                    if self.G.out_degree(child) == 0 and not all_children_leafs:
                        exists_a_leaf = True
                        break

                if all_children_leafs or not exists_a_leaf:
                    continue
                else:
                    leafs = []
                    non_leafs = []
                    for child in self.G.successors(node):
                        if self.G.out_degree(child) == 0:
                            leafs.append(child)
                        else:
                            non_leafs.append(child)
                    yield (leafs, non_leafs, node)
                
    def simple_triplets_generator(self):
        """
        Generator function that return triplets with condition: middle node has only one child
        """
        for node, degree in self.G.out_degree():
            if (
                degree >= 1
                and self.generations[node] > self.generation_depth
                and len(node) > 1
            ):
                for child in self.G.successors(node):
                    if self.G.out_degree(child) == 1:
                        yield (node, child, list(self.G.successors(child))[0])


class Collector(GeneratorMixin):
    def __init__(
        self,
        G,
        generation_depth: int = 40,
        ancestors_depth: int = 2,
        p_test=0.1,
        p_divide_leafs=0.5,
        min_to_test_rate=0.5,
    ):
        self.generation_depth = generation_depth
        self.ancestors_depth = ancestors_depth
        self.p = p_test
        self.p_divide_leafs = p_divide_leafs
        self.min_to_test_rate = min_to_test_rate
        self.G = G

        self.train = []
        self.test = []
        self.test_verteces = set()
        self.train_verteces = set()

        self.precompute_generations()

    def precompute_generations(self) -> None:
        self.generations = {}
        for i, gen in enumerate(nx.topological_generations(self.G)):
            for word in gen:
                self.generations[word] = i

    """
    Правила такие
    
    -- По только единственному ребенку если я беру вершину, которая листовая то все норм. 
    Если я беру в случай у которой только один ребенок то это тоже норм. Если у этого ребенка есть другие родители и он в других случаях попадет в тест, нас это устраивает.

    -- Если мы рассматриваем только листья, то есть два варианта:
    1) заберем половину из них в трейн, а половину в тест, то половина ушедшая в тест не должна фигурировать ни в каком другом случае в трейне. То есть мы намеренно будем фильтровать наличие этих вершин в остальных случаях. Буквально будем просматривать такие связи и выкидывать эти вершины.
    2) все эти вершины отправляются в тест только в другом кейсе применения. Они также потом должны будут фильтроваться на тест, чтобы мы их не видели в трейне. 

    -- Если мы берем случай где листья и не только листья. Мы можем отправить только листовые вершины на трейн или на тест но все вместе.

    
    """

    def collect_only_child(self):
        gen_only_child = self.only_child_generator()

        for child, parent, grandparent in gen_only_child:
            elem = {}
            elem["children"] = child
            elem["parents"] = parent
            elem["grandparents"] = grandparent
            elem["case"] = "only_child_leaf"

            if child in self.train_verteces:
                self.train.append(elem)
            else:
                if child in self.test_verteces:
                    self.test.append(elem)
                else:
                    if self.goes_to_test():
                        self.test.append(elem)
                        self.test_verteces.add(child)
                    else:
                        self.train.append(elem)
                        self.train_verteces.add(child)

    def collect_only_leafs(self):
        gen_only_leafs = self.all_children_leafs_generator()

        for parent, children in gen_only_leafs:
            elem = {}
            elem["children"] = children
            elem["parents"] = parent
            elem["grandparents"] = None
            elem["case"] = "only_leafs"

            possible_test, possible_train = self.get_possible_train(children)
            to_test_rate = len(possible_test) / len(children)

            if to_test_rate == 1:
                if self.goes_to_test():
                    if self.divide_on_half():
                        # делим пополам
                        to_test_subset = children[: len(children) // 2]
                        to_train_subset = children[len(children) // 2 :]

                        self.write_to_test(elem, to_test_subset)
                        self.write_to_train(elem, to_train_subset)

                    else:
                        self.write_to_test(elem, children)

                else:
                    self.write_to_train(elem, children)

            elif to_test_rate >= self.min_to_test_rate:
                if self.goes_to_test():
                    # эвристически пихнем больше вершин в трейн
                    to_obtain_half = (len(children) // 2) - len(possible_train)
                    possible_train = possible_train + possible_test[:to_obtain_half]
                    possible_test = possible_test[to_obtain_half:]

                    self.write_to_test(elem, possible_test)
                    self.write_to_train(elem, possible_train)

                else:
                    self.write_to_train(elem, children)
            else:
                self.write_to_train(elem, children)

    def collect_not_only_leafs(self):
        gen_not_only_leafs = self.leaf_and_no_leaf_generator()

        for children_leafs, children_no_leafs, parent in gen_not_only_leafs:
            elem = {}
            elem["children"] = children_leafs + children_no_leafs
            elem["parents"] = parent
            elem["grandparents"] = None
            elem["case"] = "leafs_and_no_leafs"

            possible_test, possible_train = self.get_possible_train(children_leafs)
            to_test_rate = len(possible_test) / len(children_leafs)
            # print(to_test_rate)
            if to_test_rate == 1:
                if self.goes_to_test():
                    self.write_to_test(elem, possible_test)
                    self.write_to_train(elem, children_no_leafs)
                else:
                    self.write_to_train(elem, children_leafs + children_no_leafs)

            else:
                self.write_to_train(elem, children_leafs + children_no_leafs)

    def collect_simple_triplets(self):
        simple_triplets = self.simple_triplets_generator()
        for grandparent, parent, child in simple_triplets:
            elem = {}
            elem["children"] = child
            elem["parents"] = parent
            elem["grandparents"] = grandparent
            elem["case"] = "simple_triplet"

            if child in self.train_verteces:
                self.train.append(elem)
            else:
                if child in self.test_verteces:
                    self.test.append(elem)
                else:
                    if self.goes_to_test():
                        self.test.append(elem)
                        self.test_verteces.add(child)
                    else:
                        self.train.append(elem)
                        self.train_verteces.add(child)

    def get_possible_train(self, children):
        """
        Returns two lists that could possibly go to test
        and that can not. It is checked through set of train verteces
        """
        possible_test = []
        possible_train = []
        for child in children:
            if child in self.train_verteces:
                possible_train.append(child)
            else:
                possible_test.append(child)

        return possible_test, possible_train

    def goes_to_test(self):
        return np.random.binomial(1, p=self.p)

    def divide_on_half(self):
        return np.random.binomial(1, p=self.p_divide_leafs)

    def write_to_test(self, elem, to_test_subset):
        """
        writes to test a subset
        """
        elem["children"] = to_test_subset
        for vertex in to_test_subset:
            self.test_verteces.add(vertex)
        self.test.append(elem)

    def write_to_train(self, elem, to_train_subset):
        """'
        writes to train verteces that are not in test
        """
        # прежде чем кладем проверяем что нет в тесте
        valid_verteces = []
        for vertex in to_train_subset:
            if vertex not in self.test_verteces:
                self.train_verteces.add(vertex)
                valid_verteces.append(vertex)

        if valid_verteces:
            elem["children"] = valid_verteces
            self.train.append(elem)


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
        )
        self.collector.collect_only_child()
        self.collector.collect_only_leafs()
        self.collector.collect_not_only_leafs()
        # if triplets needed
        self.collector.collect_simple_triplets()

        return self.collector.train, self.collector.test
