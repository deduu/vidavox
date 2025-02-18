from vidavox.document import (
    DocumentSplitter, ProcessingConfig
)

config  = ProcessingConfig()

file_path = "./unrelated/Draft_POD.md"
nodes = DocumentSplitter(config).process_file(file_path)

for node in nodes:

    print(node.metadata)




