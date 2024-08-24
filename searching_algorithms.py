
################################ Chiziqli qidiruv (Linear Search) ################################

def linear_search(arr, target):
    """
    Chiziqli qidiruv funksiyasi. Massivni boshidan oxirigacha qidiradi.
    
    :param arr: Qidiruv qilinadigan massiv.
    :param target: Qidirilayotgan qiymat.
    :return: Agar qiymat topilsa, uning indeksi; topilmasa -1.
    """
    for i in range(len(arr)):
        if arr[i] == target:
            return i  # Qiymat topildi, indeks qaytariladi
    return -1  # Qiymat topilmadi


################################ Ikkilik qidiruv (Binary Search) ################################

def binary_search(arr, target):
    """
    Ikkilik qidiruv funksiyasi. Saralangan massivda qiymatni qidiradi.
    
    :param arr: Saralangan qidiruv massiv.
    :param target: Qidirilayotgan qiymat.
    :return: Agar qiymat topilsa, uning indeksi; topilmasa -1.
    """
    low, high = 0, len(arr) - 1
    
    while low <= high:
        mid = (low + high) // 2  # O'rtacha indeksni topamiz
        if arr[mid] == target:
            return mid  # Qiymat topildi, indeks qaytariladi
        elif arr[mid] < target:
            low = mid + 1  # Qidiruv o'ng qismda davom etadi
        else:
            high = mid - 1  # Qidiruv chap qismda davom etadi
            
    return -1  # Qiymat topilmadi


################################ Kenglik birinchi qidiruv (Breadth-First Search - BFS) ################################

from collections import deque

def bfs(graph, start):
    """
    Kenglik birinchi qidiruv algoritmi. Grafning barcha tugunlarini qidiradi.
    
    :param graph: Graf (adjacency list shaklida).
    :param start: Boshlang'ich tugun.
    :return: Tugunlarning kenglik bo'yicha tartiblangan ro'yxati.
    """
    visited = set()
    queue = deque([start])
    order = []
    
    while queue:
        node = queue.popleft()  # Navbatning birinchi elementini olamiz
        if node not in visited:
            visited.add(node)  # Tugunni ko'rilgan deb belgilaymiz
            order.append(node)  # Tugunni natijaga qo'shamiz
            queue.extend(graph[node])  # Tugunning qo'shnilarini navbatga qo'shamiz
            
    return order


################################ Yagona xatolik qidiruv (Simplex Search or Hill Climbing) ################################

def hill_climbing(start, objective_function, neighbors_function):
    """
    Yagona xatolik qidiruv algoritmi. Optimal yechimga yetguncha qidiradi.
    
    :param start: Boshlang'ich nuqta.
    :param objective_function: Maqsad funktsiyasi.
    :param neighbors_function: Qo'shni nuqtalarni topuvchi funksiya.
    :return: Eng yaxshi topilgan nuqta.
    """
    current = start
    
    while True:
        neighbors = neighbors_function(current)
        next_node = max(neighbors, key=objective_function)  # Eng yaxshi qo'shnini tanlaymiz
        
        if objective_function(next_node) <= objective_function(current):
            return current  # Hech qanday qo'shni yaxshiroq emas, qidiruv to'xtaydi
        current = next_node


################################ Chuqurlik birinchi qidiruv (Depth-First Search - DFS) ################################
 
def dfs(graph, start, visited=None):
    """
    Chuqurlik birinchi qidiruv algoritmi. Grafning barcha tugunlarini qidiradi.
    
    :param graph: Graf (adjacency list shaklida).
    :param start: Boshlang'ich tugun.
    :param visited: Ko'rilgan tugunlar to'plami.
    :return: Tugunlarning chuqurlik bo'yicha tartiblangan ro'yxati.
    """
    if visited is None:
        visited = set()
    
    visited.add(start)  # Tugunni ko'rilgan deb belgilaymiz
    order = [start]
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            order.extend(dfs(graph, neighbor, visited))  # Rekursiya orqali qo'shni tugunlarni qidiramiz
            
    return order
