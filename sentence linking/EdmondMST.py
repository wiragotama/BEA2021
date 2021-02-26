from collections import defaultdict, namedtuple


Arc = namedtuple('Arc', ('tail', 'weight', 'head'))

# https://github.com/SuzeeS/Edmond-MST-Steiner-Tree-Python/blob/master/MST.py
class MSTT:
    def __init__(self):
        self.graph = [] # graph


    def add_edge(self, tail, weight, head):
        """
        tail (int): node id -- source
        weight (int or float): weight from tail to head
        head (int): node id -- target
        """
        self.graph.append(Arc(tail, weight, head))


    def spanning_arborescence(self, arcs, sink, kick):
        l=len(arcs)
        while l>=0:
            x=arcs[l-1]                                   #best incoming edges list traversed in the reverse order in which they were added
            l=l-1
            if x in kick:
               for arc in arcs:
                   if arc==kick[x] or arc in kick[x]:
                       i=arcs.index(arc)                  #every edge will kick out one edge inside the merged node, breaking the cycle.
                       arcs[i]=x                          #every edge in the Kick dictionary is removed from the best incoming edge(arcs) list
                              
        return(set(arcs))


    def find_cycle(self, successor, sink):
        visited = {sink}
        for node in successor:
            cycle = []
            while node not in visited:
                visited.add(node)           #keeping a track of the traversed nodes
                cycle.append(node)                                                               
                node = successor[node]      #if the node's parent is present in the list 'cycle',it verifies the presence of a cycle 
            if node in cycle:
                return cycle[cycle.index(node):]
        return None


    def max_spanning_arborescence(self, sink): 
        """
        sink (int): the root node id
        """
        arcs = self.graph
        kick={}                                              #dictionary storing the kicked out edges of the arcs
        good_arcs = []                                       #list storing the best (maximum) arc of every node
        quotient_map = {arc.tail: arc.tail for arc in arcs}   #dictionary for tail(key) to tail(value) mapping of every arc
        quotient_map[sink] = sink
        score={arc:arc.weight for arc in arcs}      #score for updating the weights on detecting a cycle 
        while True:
            max_arc_by_tail_rep = {}    #dictionary storing the maximum incoming arc(value) of every node(key)                                                                             
            successor_rep = {}          #dictionary storing the parent node of every node selected in max_arc_by_tail_rep                                                                       
            for arc in arcs:
                if arc.tail == sink:    #avoiding the arcs which have root as their tail
                    continue
                tail_rep = quotient_map[arc.tail]
                head_rep = quotient_map[arc.head]
                if tail_rep == head_rep:                 #avoiding the self loops 
                    continue
                if tail_rep not in max_arc_by_tail_rep or score[max_arc_by_tail_rep[tail_rep]] < score[arc]:    
                    max_arc_by_tail_rep[tail_rep] = arc         #selecting the maximum incoming arc for every node                                                       
                    successor_rep[tail_rep] = head_rep
            
            cycle_reps = self.find_cycle(successor_rep, sink)            
            if cycle_reps is None:                                 #Repeated until every non-root node has an incoming edge and no cycles are formed                    
                 z=[]                                               
                 z.extend(max_arc_by_tail_rep.values())             
                 while z:
                     y=z.pop(0)
                     good_arcs.append(y) if y not in good_arcs else None         
                 return self.spanning_arborescence(good_arcs, sink,kick)
            for arc in arcs:
                tail_rep = quotient_map[arc.tail]
                head_rep = quotient_map[arc.head]
                if arc.tail == sink:
                    continue
                if tail_rep == head_rep:
                    continue
                if tail_rep in cycle_reps and arc!= max_arc_by_tail_rep[tail_rep] and head_rep not in cycle_reps:
                    score[arc]=score[arc]-score[max_arc_by_tail_rep[tail_rep]]                                      #updating the score of the incoming edges whose tails form a cycle
                    y=kick.get(arc)
                    if y is not None:
                        kick.setdefault(arc,[]).append(max_arc_by_tail_rep[tail_rep])                               #adding the best incoming arc as the kicked out edges of every incoming arc whose score was updated
                    else:
                        kick.setdefault(arc,[]).append(max_arc_by_tail_rep[tail_rep])
            z=[]
            z.extend(max_arc_by_tail_rep.values())
            while z:
                y=z.pop(0)
                good_arcs.append(y) if y not in good_arcs else None                                                 #Adding the best incoming arc of every merged and unmerged nodes of the graph
            cycle_rep_set = set(cycle_reps)
            cycle_rep = cycle_rep_set.pop()
            quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}    #contracting the nodes in cycle_reps
            

    def min_spanning_arborescence(self, sink): 
        """
        sink (int): the root node id
        """
        arcs = self.graph
        kick={}                                              #dictionary storing the kicked out edges of the arcs
        good_arcs = []                                       #list storing the best (minimum) arc of every node
        quotient_map = {arc.tail: arc.tail for arc in arcs}   #dictionary for tail(key) to tail(value) mapping of every arc
        quotient_map[sink] = sink
        score={arc:arc.weight for arc in arcs}      #score for updating the weights on detecting a cycle 
        while True:
            min_arc_by_tail_rep = {}    #dictionary storing the minimum incoming arc(value) of every node(key)                                                                             
            successor_rep = {}          #dictionary storing the parent node of every node selected in max_arc_by_tail_rep                                                                       
            for arc in arcs:
                if arc.tail == sink:    #avoiding the arcs which have root as their tail
                    continue
                tail_rep = quotient_map[arc.tail]
                head_rep = quotient_map[arc.head]
                if tail_rep == head_rep:                 #avoiding the self loops 
                    continue
                if tail_rep not in min_arc_by_tail_rep or score[min_arc_by_tail_rep[tail_rep]] > score[arc]:    
                    min_arc_by_tail_rep[tail_rep] = arc         #selecting the maximum incoming arc for every node                                                       
                    successor_rep[tail_rep] = head_rep
            
            cycle_reps = self.find_cycle(successor_rep, sink)            
            if cycle_reps is None:                                 #Repeated until every non-root node has an incoming edge and no cycles are formed                    
                 z=[]                                               
                 z.extend(min_arc_by_tail_rep.values())             
                 while z:
                     y=z.pop(0)
                     good_arcs.append(y) if y not in good_arcs else None         
                 return self.spanning_arborescence(good_arcs, sink,kick)
            for arc in arcs:
                tail_rep = quotient_map[arc.tail]
                head_rep = quotient_map[arc.head]
                if arc.tail == sink:
                    continue
                if tail_rep == head_rep:
                    continue
                if tail_rep in cycle_reps and arc!= min_arc_by_tail_rep[tail_rep] and head_rep not in cycle_reps:
                    score[arc]=score[arc]-score[min_arc_by_tail_rep[tail_rep]]                                      #updating the score of the incoming edges whose tails form a cycle
                    y=kick.get(arc)
                    if y is not None:
                        kick.setdefault(arc,[]).append(min_arc_by_tail_rep[tail_rep])                               #adding the best incoming arc as the kicked out edges of every incoming arc whose score was updated
                    else:
                        kick.setdefault(arc,[]).append(min_arc_by_tail_rep[tail_rep])
            z=[]
            z.extend(min_arc_by_tail_rep.values())
            while z:
                y=z.pop(0)
                good_arcs.append(y) if y not in good_arcs else None                                                 #Adding the best incoming arc of every merged and unmerged nodes of the graph
            cycle_rep_set = set(cycle_reps)
            cycle_rep = cycle_rep_set.pop()
            quotient_map = {node: cycle_rep if node_rep in cycle_rep_set else node_rep for node, node_rep in quotient_map.items()}    #contracting the nodes in cycle_reps
                  
if __name__ == "__main__":                                                                                        
            
    l = {}       
    m = MSTT()
    m.add_edge(1,1,3)
    m.add_edge(3,1,2)
    m.add_edge(2,1,1)
    m.add_edge(3,5,1)
    m.add_edge(1,5,2)
    m.add_edge(2,5,3)
    l=m.min_spanning_arborescence(1)              #Graph 1 :Graph with more than one cycles 
    print ('\n\n',l)
#l=m.max_spanning_arborescence([Arc(1,5,0),Arc(2,1,0),Arc(3,1,0),Arc(2,11,1),Arc(3,4,1),Arc(3,5,2),Arc(1,10,2),Arc(1,9,3),Arc(2,8,3)],0) #Graph 2:Graph with more than one cycles        
#l=m.max_spanning_arborescence([Arc(2 ,1,1),Arc(3,3,2),Arc(5,4,3),Arc(6,3,5),Arc(5,2,4),Arc(5,1,2),Arc(4,1,7),Arc(1,2,4),Arc(4,1,3)],7)  #Graph 3:Changing the root node of Graph 1 gives us a different spanning arborescence          
#l=m.max_spanning_arborescence([Arc(2,5,1),Arc(1,-2,2),Arc(2,-3,3),Arc(4,-4,1),Arc(2,7 ,4),Arc(4,9,3),Arc(3,8,1),Arc(1,6,0),Arc(3,7,0)],0)#Graph 4:Graph with negative edges
#l=m.max_spanning_arborescence([Arc(1, 9, 0), Arc(2, 10, 0), Arc(3, 9, 0), Arc(2, 20, 1), Arc(3, 3, 1), Arc(1, 30, 2), Arc(3, 11, 2), Arc(2, 0, 3), Arc(1,30, 3)], 0)#Graph 5:On reversing all the edges of the graph we do not get a spanning arborescence
#l=m.max_spanning_arborescence([Arc(1,5,0),Arc(2,5,0),Arc(3,5,0),Arc(4,5,0),Arc(0,6,1),Arc(2,6,1),Arc(3,6,1),Arc(4,6,1),Arc(0,7,2),Arc(1,7,2),Arc(3,7,2),Arc(4,7,2),Arc(0,8,3),Arc(1,8,3),Arc(2,8,3),Arc(4,8,3),Arc(0,9,4),Arc(1,9,4),Arc(2,9,4),Arc(3,9,4)],0) #Graph 5 : Complete graph of 5 nodes              
#l=m.max_spanning_arborescence([Arc(0,10,0),Arc(1,9,1),Arc(2,10,2)],0)  #Graph 6 : Fails to give an arborescence for a null or disconnected graph
#l=m.max_spanning_arborescence([Arc(1,9,0),Arc(2,8,0),Arc(2,10,1),Arc(1,1,2),Arc(2,6,3),Arc(3,7,2),Arc(1,12,3)],0) #Graphs with varying edge weights 
#l=m.max_spanning_arborescence([Arc(1,18,0),Arc(2,9,0),Arc(2,10,1),Arc(1,50,2)],0)  
#l=m.max_spanning_arborescence([Arc(1,5,0),Arc(2,1,0),Arc(3,1,0),Arc(2,11,1),Arc(3,4,1),Arc(3,50,2),Arc(1,100,2),Arc(1,9,3),Arc(2,8,3)],0)
#l=m.max_spanning_arborescence([Arc(1,5,0),Arc(2,1,0),Arc(3,1,0),Arc(2,110,1),Arc(3,4,1),Arc(3,5,2),Arc(1,10,2),Arc(1,9,3),Arc(2,8,3)],0)
#l=m.max_spanning_arborescence([Arc(1,1,0),Arc(2,2,0),Arc(3,3,0),Arc(5,4,0),Arc(2,100,1),Arc(5,7,4),Arc(4,10,3),Arc(5,5,2),Arc(2,6,5),Arc(3,11,2)],0)  
#l=m.max_spanning_arborescence([Arc(1,4,0),Arc(2,5,0),Arc(3,6,0),Arc(4,7,0),Arc(1,10,2),Arc(2,20,1),Arc(3,1,2),Arc(4,30,3),Arc(3,20,4)],0)  