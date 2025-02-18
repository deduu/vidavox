

class Store:
    def __init__(self, index):
        self.index = index
        self.index.add_documents(self.documents)
        self.index.add_documents(self.metadata)
