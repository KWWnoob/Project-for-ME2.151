'''
Finger Forward Dyanmics V1
'''

class Finger:

    def __init__(self, routing) -> None:
        self.routing = routing

    class Section:

        def __init__(self, orientation, starting_node, ending_node) -> None:
            self.orientation = orientation
            self.starting_node = starting_node
            self.ending_node = ending_node
            