from collections import defaultdict, Counter
class BeautifulGraph():
    def __init__(self, n,m,word, source, target):
        self.nodes = list(range(1,n+1))
        self.edges = m
        self.word = word
        self.neighbours = defaultdict(list)
        for s,t in zip(source, target):
            self.neighbours[s].append(t)
        self.alphabet = dict(zip(self.nodes, self.word))

    def node_words(self, node, visited, recStack, initial_word = "", word_list = []):
        initial_word += self.alphabet[node]
        cyclic = False
        visited[node] = True
        recStack[node] = True 
        for neighbor in self.neighbours[node]:
            if not visited[neighbor]:    
                cyclic, word_list = self.node_words(neighbor, visited, recStack, initial_word=initial_word, word_list=word_list)
                if cyclic:
                    return cyclic,[]
            elif recStack[neighbor]:
                return True, []
        if not self.neighbours[node]:
            word_list.append(initial_word)
        recStack[node] = False
        return  cyclic, word_list

    def beauty(self):
        max_beauty = -1
        visited = defaultdict(bool)
        recStack = defaultdict(bool)
        for node in self.nodes:
            cyclic, word_list = self.node_words(node, visited, recStack)
            if cyclic:
                return -1
            max_beauty = max(max_beauty, self.max_repeated(word_list))
        return max_beauty

    @staticmethod
    def max_repeated(word_list):
        return max(map(lambda word: max(Counter(word).values()), word_list))
            
        
#print(BeautifulGraph(5,4,"abaca", [1,1,3,4],[2,3,4,5]).__dict__)

#graph1 = BeautifulGraph(5,4,"abaca", [1,1,3,4],[2,3,4,5])
#print(graph1.max_repeated(graph1.node_words(1, [],[])))
#print(graph1.beauty())

#print(graph2.beauty())