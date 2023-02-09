from beauty import BeautifulGraph
def test_1():
    graph1 = BeautifulGraph(5,4,"abaca", [1,1,3,4],[2,3,4,5])
    assert graph1.beauty() == 3

def test_2():
    graph2 = BeautifulGraph(6,5,"xyzabc", [1,2,3,4,5,6],[2,3,1,3,4,4])
    assert graph2.beauty() == -1

if __name__ == "__main__":
    test_1()
    test_2()