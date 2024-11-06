import heapq


class PriorityQueue:
    def __init__(self, key=lambda x: x, mode="min"):
        self.heap = []
        self.mode = mode

        if mode == "max":
            self.key = lambda x: -key(x)
        elif mode == "min":
            self.key = key

    def push(self, item):
        heapq.heappush(self.heap, (self.key(item), item))

    def pop(self):
        return heapq.heappop(self.heap)[1]

    def __len__(self):
        return len(self.heap)

    def clear(self):
        self.heap.clear()

    def empty(self):
        return len(self.heap) == 0