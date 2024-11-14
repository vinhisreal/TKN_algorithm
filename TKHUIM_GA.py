import random
#lay dataset tu file
transactions= [
    {
        "TID": "T1",
        "items": ["a", "b", "c"],
        "quantities": [1, 3, 2],
        "profit": [5, 10, 2]
    },
    {
        "TID": "T2",
        "items": ["b", "c"],
        "quantities": [1,2],
        "profit": [10,2]
    },
    {
        "TID": "T3",
        "items": ["a","c","d"],
        "quantities": [1,2,4],
        "profit": [5,2,1]
    },
    {
        "TID": "T4",
        "items": ["a", "b", "d"],
        "quantities": [1,2,4],
        "profit": [5,10,1]
    }
]


# def get_dataset():
#     data = []
#     with open('dataset.txt', 'r', encoding='utf-8') as dataset:
#         next(dataset)  
#         for line in dataset:
#             parts = line.split()
#             transaction_id = parts[0]
#             items = [str(u) for u in parts[1].split(',')] 
#             quantity = [int(u) for u in parts[2].split(',')]
#             profit = [int(u) for u in parts[3].split(',')]
#             data.append([[transaction_id], items, quantity, profit])
#     return data

def get_utility(data):
    unit = set()
    for trans in data:
        for item in trans[1]:
            unit.add(item)
    utility = []
    for item in unit:
        utility.append([item, 0])
    return utility
#nhan utility vao dataset
def calculate_utility(data, utility):
    dataset=[]
    for i in range(len(data)):
        dataset.append([data[i][0],data[i][1],data[i][2]])
        for j in range(len(data[i][1])):
            dataset[i][2][j] = data[i][2][j]*data[i][3][j]
    return dataset

def utility_itemset(dataset, utility):
    utility_trans=utility

    for k in range(len(utility_trans)):
        utility_trans[k][1]=0
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])):
            for k in range(len(utility_trans)):
                if dataset[i][1][j] == utility_trans[k][0]:
                    utility_trans[k][1] += dataset[i][2][j]
    return utility_trans


# data = transactions
# utility = list(get_utility(data))
# dataset = calculate_utility(data, utility)
#tinh utility cua tung gia tri trong toan bo dataset

# u=utility_itemset(dataset,utility)
#tinh utility cua tung trans
def TU(transaction):
    # Calculate the total utility as the sum of profit * quantity for each item in the transaction
    return sum(q * p for q, p in zip(transaction["quantities"], transaction["profit"]))

def get_top_m_items(transaction, m):
    # Calculate utility for each item in the transaction
    item_utilities = [(item, q * p) for item, q, p in zip(transaction["items"], transaction["quantities"], transaction["profit"])]
    # Sort items by utility in descending order and return the top m items
    top_items = sorted(item_utilities, key=lambda x: x[1], reverse=True)[:m]
    # Return items and their utilities
    return [item[0] for item in top_items], [item[1] for item in top_items]

#althogirm 1

def initial_solutions(dataset, n, m):
    trans_P = []
    P = []
    for transaction in dataset:
        u = TU(transaction)
        X_items, X_utilities = get_top_m_items(transaction, m)
        
        if len(P) < n:
            trans_P.append((X_items, X_utilities))
            P.append(X_utilities)
        else:
            # Find transaction with the minimum utility in trans_P
            min_utility_trans = min(trans_P, key=lambda t: sum(t[1]))
            min_utility = sum(min_utility_trans[1])
            
            # Replace if current transaction has higher utility
            if u > min_utility:
                trans_P.remove(min_utility_trans)
                P.remove(min_utility_trans[1])
                trans_P.append((X_items, X_utilities))
                P.append(X_utilities)
    return P


print(initial_solutions(transactions,2,2))
# def F(X):
#     sum=0
#     for i in range(len(dataset)):
#         if set(X).issubset(set(dataset[i][1])):
#             for j in range(len(dataset[i][1])):
#                 if dataset[i][1][j] in X:
#                     sum+= dataset[i][2][j]
#     return sum 

# def roullete_wheel(utility_itemset):
#     elements=[]
#     weights=[]
#     sum=0
#     for item in utility_itemset:
#         elements.append(item[0])
#         weights.append(item[1])
#         sum+=item[1]
#     for i in range(len(weights)):
#         weights[i]=weights[i]/sum
#     return random.choices(elements,weights,k=1)
# def genetic_operators(S,a,b):
#     P=[]
#     for i in range(len(S)):
#         for j in range(i + 1, len(S)):
#             Xi=S[i]
#             Xj=S[j]
#             if a>random.uniform(0,1):
#                 x=''
#                 y=''
#                 if F(Xi)>F(Xj):
#                     minXi=0
#                     maxXj=0
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xi:
#                             minXi=item[1]
#                             x=item[0]
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xi:
#                             if(item[1]<minXi):
#                                 minXi=item[1]
#                                 x=item[0]
                                
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xj:
#                             maxXj=item[1]
#                             y=item[0]
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xj:
#                             if(item[1]>maxXj):
#                                 maxXj=item[1]
#                                 y=item[0]
#                 else:
#                     minXj=0
#                     maxXi=0
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xj:
#                             minXj=item[1]
#                             y=item[0]
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xj:
#                             if(item[1]<minXj):
#                                 minXj=item[1]
#                                 y=item[0]
                                
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xi:
#                             maxXi=item[1]
#                             x=item[0]
#                     for item in utility_itemset(dataset,utility):
#                         if item[0] in Xi:
#                             if(item[1]>maxXi):
#                                 maxXi=item[1]
#                                 x=item[0]
#                 Xi=set(Xi)-{x}
#                 Xi=set(Xi)|{y}
#                 Xj = set(Xj) - {y} 
#                 Xj = set(Xj) | {x} 
#             for X in [Xi,Xj]:
#                 if b>random.uniform(0,1):
#                     x=''
#                     if 0.5>random.uniform(0,1):
#                         minX=0
#                         for item in utility_itemset(dataset,utility):
#                             if item[0] in X:
#                                 minX=item[1]
#                                 x=item[0]
#                         for item in utility_itemset(dataset,utility):
#                             if item[0] in X:
#                                 if(item[1]<minX):
#                                     minX=item[1]
#                                     x=item[0]
#                         X=set(X)-{x}
#                     else:
#                         x=roullete_wheel(utility_itemset(dataset,utility))
#                         X=set(X)|set(x)
#                 P.append(X)
#     return P   
# def printdata(x):
#     for item in x:    
#         print(item)                
# # printdata(data)
# # printdata(utility)
# # printdata(utility_itemset(data,utility))
# # print(initial_solutions(data,2,2))
# def TKHUIM_GA(dataset,n,m,e):
#     P=[]
#     E=[]
#     u=utility_itemset(dataset,utility)
#     u.sort(key=lambda x: x[1], reverse=True)

#     # Lấy m phần tử đầu tiên
#     top_m_items = u[:e]

#     # Tách các mục và giá trị utility
#     top_items = [item[0] for item in top_m_items]
#     E.append(top_items)
    
#     exit=False
#     P=initial_solutions(dataset,n,m)
#     a=0.5
#     b=0.5
#     while True:
#         S=TournamentSelection(P,len(P)-1,n)
        
#         P=genetic_operators(S,a,b)
#         result = []
#         for sublist in P:
#             for element in E:
#                 combined_set = set(sublist) | set(element)  # Hợp nhất các phần tử và loại bỏ trùng lặp
#                 result.append(list(combined_set))
#         result.sort(key=F, reverse=True)
#         new_E = result[:e]
#         if new_E != E:
#             a=a+0.05
#             b=b-0.05
#             E = new_E
#         else:
#             a=a-0.05
#             b=b+0.05
        
#         if round(b,2)==1.00:
#             exit=True
#         if exit:
#             break
#     return E
# def Tournament(T, k):
#     # INPUT
#     #   T = a list of individuals randomly selected from a population.
#     #   k = the tournament size. In other words, the number of elements in T.
#     # OUTPUT
#     #   the fittest individual.
#     Best=T[0]
#     for i in range(1,k):
#         Next=T[i]
#         if F(Next) > F(Best):
#             Best = Next
#     return Best
# #Assume we wish to select n individuals from the population P
# def TournamentSelection(P, k, n):
#     # INPUT
#     #   P = the population.
#     #   k = the tournament size, such that 1 ≤ k ≤ the number of individuals in P.
#     #   n = the total number of individuals we wish to select.
#     # OUTPUT
#     #   the pool of individuals selected in the tournaments.
#     T = []
#     B = [None] * n
#     for i in range(n):
#         # Pick k individuals from P at random, with or without replacement, and add them to T
#         T = random.choices(P, k=k)
#         B[i] = Tournament(T, len(T))
#         T = [ ]
#     return B
# print(TKHUIM_GA(dataset,4,5,6))