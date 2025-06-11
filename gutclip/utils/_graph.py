import numpy as np
import os
import tempfile
class Node:
    def __init__(self, node):
        self.id = node
        self.parent = None
        self.children = []
        self.abundance = 0
        self.layer = 0

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_id(self):
        return str(self.id)

    def set_id(self, id):
        self.id = id

    def set_parent(self, parent):
        self.parent = parent
        self.layer = parent.layer + 1

    def get_layer(self):
        return self.layer

    def set_layer(self, layer):
        self.layer = layer

    def get_abundance(self):
        return self.abundance

    def set_abundance(self, val):
        self.abundance = val

    def calculate_abundance(self):
        self.abundance = sum([c.abundance for c in self.children])

    def get_leaves(self):
        count = 0
        stack = self.children.copy()
        while stack:
            node = stack.pop()
            if not node.children:
                count += 1
            else:
                stack.extend(node.children)
        return count


class Graph:
    def __init__(self):
        self.nodes = []
        self.layers = 0
        self.width = 0
        self.root = None
        self.NODE_DICT = {}
        self.node_count = 0

    def add_node(self, layer, node):
        assert layer >= 0, f"Layer must be non-negative, got {layer}"
        while len(self.nodes) <= layer:
            self.nodes.append({})
        self.nodes[layer][node] = node
        self.NODE_DICT[str(node.get_id())] = node
        self.layers = max(self.layers, layer + 1)

    def get_nodes(self, layer):
        return list(self.nodes[layer].keys()) if layer < len(self.nodes) else []

    def build_graph(self, newick_file):
        with open(newick_file) as f:
            newick = f.read().strip().rstrip(';')

        stack = []
        current_id = 0
        layer = 0
        node = None
        token = ""

        for char in newick:
            if char == '(':
                if token.strip():
                    leaf = Node(token.strip())
                    leaf.set_layer(layer)
                    stack[-1].append(leaf)
                    token = ""
                stack.append([])
                layer += 1
            elif char == ',':
                if token.strip():
                    leaf = Node(token.strip())
                    leaf.set_layer(layer)
                    stack[-1].append(leaf)
                    token = ""
            elif char == ')':
                if token.strip():
                    leaf = Node(token.strip())
                    leaf.set_layer(layer)
                    stack[-1].append(leaf)
                    token = ""
                children = stack.pop()
                layer -= 1
                parent = Node(f"inner_{current_id}")
                current_id += 1
                parent.set_layer(layer)
                for child in children:
                    child.set_parent(parent)
                    parent.add_child(child)
                if stack:
                    stack[-1].append(parent)
                else:
                    node = parent
            else:
                token += char

        self.root = node
        self._collect_nodes_by_layer(self.root)
        self.width = sum(n.get_leaves() for n in self.get_nodes(0))

    def _collect_nodes_by_layer(self, root):
        def dfs(node):
            layer = node.get_layer()
            while len(self.nodes) <= layer:
                self.nodes.append({})
            self.nodes[layer][node] = node
            self.NODE_DICT[str(node.get_id())] = node
            self.layers = max(self.layers, layer + 1)
            for child in node.get_children():
                dfs(child)

        self.nodes = []
        self.layers = 0
        self.NODE_DICT = {}
        if root:
            dfs(root)
    def build_graph_from_string(self, newick_str: str):
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write(newick_str)
            f.flush()
            self.build_graph(f.name)
        os.remove(f.name)

    # def populate_graph(self, feature_labels, abundance_vector):
    #     layer = self.layers - 1
    #     tracker = {f: False for f in feature_labels}

    #     while layer >= 0:
    #         for node in self.get_nodes(layer):
    #             if node.children:
    #                 total = sum([child.get_abundance() for child in node.children])
    #                 for i, f in enumerate(feature_labels):
    #                     if node.get_id() == f:
    #                         total += abundance_vector[i]
    #                         tracker[f] = True
    #                 node.set_abundance(total)
    #             else:
    #                 for i, f in enumerate(feature_labels):
    #                     if node.get_id() == f:
    #                         node.set_abundance(abundance_vector[i])
    #                         tracker[f] = True
    #         layer -= 1
    def populate_graph(self, feature_labels, abundance_vector):
        for node in self.NODE_DICT.values():
            node.set_abundance(0.0)

        missing_features = []

        for i, f in enumerate(feature_labels):
            if f in self.NODE_DICT:
                self.NODE_DICT[f].set_abundance(abundance_vector[i])
            else:
                missing_features.append(f)

        if missing_features:
            print(f"[Warning] {len(missing_features)} OTUs not found in tree.")
            print("Missing OTUs:", missing_features)


        # 自底向上计算内部节点
        for layer in reversed(range(self.layers)):  # 从最深层开始
            for node in self.get_nodes(layer):
                if node.children:  # 仅处理内部节点
                    total = sum(child.abundance for child in node.children)
                    node.set_abundance(total)

    def get_map(self, permute=-1):
        m = np.zeros((self.layers, self.width))
        current = self.get_nodes(0)
        for i in range(self.layers):
            j = 0
            next_nodes = []
            for node in current:
                if j >= self.width:
                    break
                m[i][j] = node.get_abundance()
                children = node.get_children()
                if permute >= 0:
                    np.random.seed(permute)
                    np.random.shuffle(children)
                next_nodes.extend(children)
                j += 1
            current = next_nodes
        return m
    def get_node_by_id(self, node_id):
        # node_id 可以为 str 或非str
        sid = str(node_id)
        return self.NODE_DICT.get(sid, None)