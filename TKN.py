import heapq

# Define dataset and profits
TKHQ = []
# transactions = {
#     "t1": {"a": 1, "b": 2, "c": 2, "d": 1},
#     "t2": {"a": 1, "b": 3, "c": 3, "d": 2, "e": 2},
#     "t3": {"a": 1, "c": 6, "e": 3},
#     "t4": {"b": 3, "d": 5, "e": 2},
#     "t5": {"b": 1, "c": 5, "d": 1, "e": 4},
#     "t6": {"c": 2, "d": 1, "e": 1},
# }

# profits = {"a": 4, "b": 3, "c": 1, "d": -1, "e": 2}

# transactions={
#     't1': {'a': 2, 'b': 2, 'd': 1, 'e': 3, 'f': 2, 'g': 1},
#     't2': {'b': 1, 'c': 5},
#     't3': {'b': 2, 'c': 1, 'd': 3, 'e': 2, 'f': 1},
#     't4': {'c': 2, 'd': 1, 'e': 3},
#     't5': {'a': 2, 'f': 3},
#     't6': {'a': 2, 'b': 1, 'c': 4, 'd': 2, 'e': 1, 'f': 3, 'g': 1},
#     't7': {'b': 3, 'c': 2, 'e': 2}
# }
# profits={
#     'a': 4, 'b': 3, 'c': 1, 'd': -1, 'e': 2, 'f': -1, 'g': -2
# }

# transactions={
#     't1': {'a': 2, 'b': 3, 'd': 1, 'h': 1},
#     't2': {'a': 2, 'c': 4, 'e': 2, 'h': 3},
#     't3': {'b': 6, 'c': 3, 'd': 1, 'e': 3, 'f': 2},
#     't4': {'a': 4, 'b': 3, 'c': 3, 'g': 2},
#     't5': {'b': 4, 'd': 4, 'e': 1, 'g': 2, 'h': 1}
# }

# profits={
#     'a': 2, 'b': 1, 'd': 3, 'h': -1, 'c': 1, 'e': -1, 'f': 5, 'g': -1
# }


def read_data(file_name: str) -> dict:
    """
    Read and parse data from a given file.

    Parameters:
    file_name (str): The name of the file to read data from. The default value is "table.txt".

    Returns:
    dict: The parsed data from the file. The data is expected to be in a format that can be evaluated as a Python dictionary.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    SyntaxError: If the data in the file cannot be evaluated as a Python dictionary.
    """
    with open(file_name, "r") as file:
        data = file.read()

    dataset = eval(data)
    return dataset


def process_data(dataset: list) -> tuple:
    """
    Process a dataset to extract transactions and item profits.

    Parameters:
    dataset (list): A list of dictionaries, where each dictionary represents a transaction.
                    Each transaction dictionary contains the following keys:
                    - "TID": Transaction ID
                    - "items": A list of item IDs in the transaction
                    - "quantities": A list of quantities for each item in the transaction
                    - "profit": A list of profit values for each item in the transaction

    Returns:
    tuple: A tuple containing two dictionaries:
           - transactions: A dictionary representing the transactions, where keys are transaction IDs
                           and values are dictionaries of items and their quantities.
           - profits: A dictionary representing the profit values for each item, where keys are item IDs
                      and values are the corresponding profit values.
    """
    # Initialize dictionaries for transactions and profits
    transactions = {}
    profits = {}

    for entry in dataset:
        tid = entry["TID"].lower()  # Ensure TID is always in lowercase
        items = entry["items"]
        quantities = entry["quantities"]
        profit_list = entry["profit"]

        # Create dictionary for transactions
        transactions[tid] = {item: qty for item, qty in zip(items, quantities)}

        # Update profits: only take the first profit value of each item from each transaction
        for item, profit in zip(items, profit_list):
            if item not in profits:
                profits[item] = (
                    profit  # Assign profit to item if it's not already in profits
                )

    return transactions, profits


# READ
data = read_data("data.txt")
transactions, profits = process_data(data)


def calculate_priu(transactions: dict, profits: dict) -> dict:
    """
    Calculate the Positive Item Utility (PRIU) for each item in the transactions.

    PRIU is calculated as the sum of the product of the quantity and profit for each item in a transaction.

    Parameters:
    transactions (dict): A dictionary of transactions, where each transaction is a dictionary of items and their quantities.
    profits (dict): A dictionary of profits for each item.

    Returns:
    dict: A dictionary of PRIU values for each item.
    """
    priu_dict = {}
    for transaction in transactions.values():
        for item, quantity in transaction.items():
            utility = quantity * profits.get(
                item, 0
            )  # calculate utility for each item in transaction
            if item in priu_dict:
                priu_dict[item] += utility
            else:
                priu_dict[item] = utility
    return priu_dict


def LIUS_Build_contiguous(sorted_database: list, priu_list: dict) -> list:
    """
    Build a LIUS matrix for contiguous itemsets in a sorted transaction database.

    Parameters:

    sorted_database (list): A list of transactions, where each transaction is a dictionary of items and their utilities.

    priu_list (dict): A dictionary of PRIU values for all positive items x R g.

    Returns:
    list: A LIUS matrix, represented as a list of lists. Each element in the matrix is a dictionary containing the contiguous sequence and its utility.
    """
    # Sort the positive items in a predefined order
    items = list(priu_list.keys())
    n = len(items)

    # Initialize a triangular matrix with None values
    lius_matrix = [[None for _ in range(n)] for _ in range(n)]

    # Fill the cumulative utility of contiguous itemsets into the triangular matrix
    for transaction in sorted_database:
        for i in range(n):
            cumulative_utility = 0
            for j in range(
                i + 1, n
            ):  # Start from i + 1 to ensure the sequence length is >= 2
                # Contiguous sequence from items[i] to items[j]
                contiguous_sequence = items[i : j + 1]

                # Calculate the utility of the contiguous sequence in the current transaction
                utility = calculate_utility_of_itemset(transaction, contiguous_sequence)
                cumulative_utility += utility

                # If the matrix cell has no value, initialize it with the contiguous sequence and its initial utility
                if lius_matrix[i][j] is None:
                    lius_matrix[i][j] = {"sequence": contiguous_sequence, "utility": 0}

                # Add to the utility of the contiguous sequence
                lius_matrix[i][j]["utility"] += utility

    return lius_matrix


def priu_strategy(priu_dict: dict, k: int) -> float:
    """
    Implements the PRIU strategy for utility threshold determination.

    The PRIU strategy selects the k-th highest positive item utility value from the sorted list of PRIU values.
    If k is greater than the number of positive items, the function returns the highest utility value.

    Parameters:
    priu_dict (dict): A dictionary containing item names as keys and their corresponding PRIU values as values.
    k (int): The position of the utility value to be selected.

    Returns:
    float: The utility threshold determined by the PRIU strategy.
    """
    priu_values = sorted(
        priu_dict.values(), reverse=True
    )  # Sort PRIU values in descending order
    min_util = (
        priu_values[k - 1] if k <= len(priu_values) else priu_values[-1]
    )  # Select the k-th highest value or the highest value if k is out of range
    return min_util


def PLIU_E_strategy(lius_matrix: list, minUtil: float, k: int) -> tuple:
    """
    Perform the PLIU-E strategy to update the minimum utility threshold and maintain a priority queue.

    Parameters:
    lius_matrix (list): A 2D list representing the LIUS matrix, where each element is a dictionary containing utility information.
    minUtil (float): The current minimum utility threshold.
    k (int): The maximum number of elements to maintain in the priority queue.

    Returns:
    tuple: A tuple containing the updated minimum utility threshold and the priority queue (PIQULIU).
    """

    # Initialize the priority queue PIQULIU with a maximum size of k
    PIQULIU = []

    # Iterate through the LIUS matrix to add utility values to PIQULIU
    for i in range(len(lius_matrix)):
        for j in range(i + 1, len(lius_matrix)):
            if (
                lius_matrix[i][j] is not None
            ):  # Check if the matrix cell contains a value
                utility = lius_matrix[i][j]["utility"]

                # Only add to PIQULIU if utility >= minUtil
                if utility >= minUtil:
                    if len(PIQULIU) < k:
                        # If the queue is not yet at its maximum size, add directly
                        heapq.heappush(PIQULIU, utility)
                    else:
                        # If the queue is at its maximum size, replace the smallest value if utility is larger
                        heapq.heappushpop(PIQULIU, utility)

    # If PIQULIU has at least k elements, update minUtil to the smallest value in PIQULIU
    if len(PIQULIU) == k:
        # The smallest value in PIQULIU is the kth largest value, so use it to update minUtil
        new_minUtil = PIQULIU[0]
        if new_minUtil > minUtil:
            minUtil = new_minUtil

    return minUtil, PIQULIU


def PLIU_LB_strategy(
    lius_matrix: list, priu_list: dict, piqu_liu: list, k: int, minUtil: float
) -> float:
    """
    Calculates the minimum utility threshold using a lower bound strategy based on the LIUS matrix and a priority queue.

    Parameters:
    lius_matrix (list): A 2D matrix of utility values for item sequences.
    priu_list (dict): A dictionary of item utilities.
    piqu_liu (list): A list of previously calculated utility values.
    k (int): The number of largest utility values to consider.
    minUtil (float): The current minimum utility value.

    Returns:
    float: The updated minimum utility value after considering the k largest utilities.
    """
    # Create a new priority queue to store lower bound utility values
    piqu_lb_liu = []

    # Iterate through all pairs in the LIUS matrix
    for i in range(len(lius_matrix)):
        for j in range(i + 1, len(lius_matrix)):
            if lius_matrix[i][j] is not None:
                util_base = lius_matrix[i][j]["utility"]

                # Items from the start to the end of the sequence
                pos_start_item = i + 1
                pos_end_item = j - 1

                # Iterate through the items between pos_start_item and pos_end_item
                for x in range(pos_start_item, pos_end_item + 1):
                    # Access the utility of item x by its name
                    item_x = list(priu_list.keys())[x]
                    util_lb = util_base - priu_list[item_x]

                    # If the utility is greater than minUtil, add it to the priority queue
                    if util_lb > minUtil:
                        heapq.heappush(piqu_lb_liu, util_lb)

                    # Continue removing item y
                    for y in range(x + 1, pos_end_item + 1):
                        item_y = list(priu_list.keys())[y]
                        util_lb = util_base - priu_list[item_x] - priu_list[item_y]

                        if util_lb > minUtil:
                            heapq.heappush(piqu_lb_liu, util_lb)

                        # Continue removing item z
                        for z in range(y + 1, pos_end_item + 1):
                            item_z = list(priu_list.keys())[z]
                            util_lb = (
                                util_base
                                - priu_list[item_x]
                                - priu_list[item_y]
                                - priu_list[item_z]
                            )

                            if util_lb > minUtil:
                                heapq.heappush(piqu_lb_liu, util_lb)

                            # Remove item w
                            for w in range(z + 1, pos_end_item + 1):
                                item_w = list(priu_list.keys())[w]
                                util_lb = (
                                    util_base
                                    - priu_list[item_x]
                                    - priu_list[item_y]
                                    - priu_list[item_z]
                                    - priu_list[item_w]
                                )

                                if util_lb > minUtil:
                                    heapq.heappush(piqu_lb_liu, util_lb)

    # Create PIQU_ALL as the union of piqu_liu and piqu_lb_liu
    piqu_all = piqu_liu + piqu_lb_liu
    piqu_all = heapq.nlargest(k, piqu_all)

    # If there are enough k values in piqu_all and the k-th largest value is greater than minUtil, update minUtil
    if len(piqu_all) >= k:
        kth_largest_value = piqu_all[k - 1]
        if kth_largest_value > minUtil:
            minUtil = kth_largest_value

    return minUtil


def calculate_utility(quantity: int, profit: float) -> float:
    """
    Calculate the utility of a given item based on its quantity and profit.

    Parameters:
    quantity (int): The quantity of the item.
    profit (float): The profit generated by selling one unit of the item.

    Returns:
    float: The total utility generated by selling the given quantity of the item.
    """
    return quantity * profit


def calculate_ptwu(transactions: dict, profit: dict) -> dict:
    """
    Calculate the Positive Transactional Utility (PTWU) for each item in the transactions.

    PTWU is calculated as the sum of the product of the quantity and profit for each item in a transaction,
    where the profit is greater than zero.

    Parameters:
    transactions (dict): A dictionary of transactions, where each transaction is a dictionary of items and their quantities.
    profit (dict): A dictionary of profits for each item.

    Returns:
    dict: A dictionary of PTWU values for each item.
    """
    PTWU = {}
    for transaction_id, items in transactions.items():
        for item, quantity in items.items():
            if item not in PTWU:
                PTWU[item] = 0
    for item in PTWU:
        for transaction_id, items in transactions.items():
            if any(i[0] == item for i, quantity in items.items()):
                for i, quantity in items.items():
                    profit = profits.get(i)
                    if profit > 0:
                        PTWU[item] += calculate_utility(quantity, profit)
    return {item: ptwu for item, ptwu in PTWU.items()}


def compute_priu_list(transactions: dict, profits: dict) -> dict:
    """
    Compute the Positive Item Utility (PRIU) list for a given transaction database.

    Parameters:
    transactions (dict): A dictionary representing the transaction database, where keys are transaction IDs and values are dictionaries of items and their quantities.
    profits (dict): A dictionary representing the profit values for each item, where keys are item IDs and values are the corresponding profit values.

    Returns:
    dict: A dictionary representing the PRIU list, where keys are item IDs and values are the corresponding PRIU values. The PRIU list only includes items with positive PRIU values and excludes items in the set of promising negative items (g).
    """
    priu_dict = calculate_priu(transactions, profits)  # Get PRIU values for all items
    g = {
        item for item, utility in priu_dict.items() if utility < 0
    }  # Set of promising negative items

    # Filter out items in g, keeping only those with positive PRIU values (PRIU >= 0)
    priu_list = {
        item: priu for item, priu in priu_dict.items() if priu >= 0 and item not in g
    }

    return priu_list


def sort_secondary_a_by_total_order(
    secondary_a: list, ptwu: dict, profits: dict
) -> list:
    """
    Sort the items in secondary_a based on their total order.

    The function separates the items into positive and negative groups based on their profit.
    It then sorts the positive items in ascending order based on their Ptwu values,
    and the negative items in ascending order based on their Ptwu values.
    Finally, it combines the sorted positive and negative items to create the sorted secondary_a.

    Parameters:
    secondary_a (list): A list of items to be sorted.
    ptwu (dict): A dictionary containing the Ptwu values for each item.
    profits (dict): A dictionary containing the profit values for each item.

    Returns:
    list: A sorted list of items based on their total order.
    """
    # Split the items into positive and negative groups based on profit
    positive_items = {
        item: ptwu[item] for item in secondary_a if profits.get(item, 0) > 0
    }
    negative_items = {
        item: ptwu[item] for item in secondary_a if profits.get(item, 0) < 0
    }

    # Sort the positive items in ascending order by Ptwu
    sorted_positive = sorted(positive_items, key=lambda x: positive_items[x])

    # Sort the negative items in ascending order by Ptwu
    sorted_negative = sorted(negative_items, key=lambda x: negative_items[x])

    # Combine the sorted positive and negative items
    sorted_secondary_a = sorted_positive + sorted_negative
    return sorted_secondary_a


def filter_transactions_by_secondary_a(D: dict, sorted_secondary_a: list) -> dict:
    """
    Filter transactions in D based on the presence of items in sorted_secondary_a.

    Parameters:
    D (dict): A dictionary of transactions, where each transaction is a dictionary of items and their quantities.
    sorted_secondary_a (list): A list of items sorted according to a specific order.

    Returns:
    dict: A dictionary of filtered transactions, where each transaction is a dictionary of items and their quantities.
          Only transactions containing items from sorted_secondary_a are included in the result.
    """
    # Convert Secondary(a) from a list to a set for efficient membership testing
    secondary_set = set(sorted_secondary_a)

    # Create an empty dictionary to store the filtered transactions
    filtered_transactions = {}

    # Iterate over each transaction in D
    for transaction_id, items in D.items():
        # Filter items in the current transaction to keep only those in secondary_set
        filtered_items = {
            item: quantity for item, quantity in items.items() if item in secondary_set
        }

        # Only add the transaction to the filtered_transactions dictionary if it is not empty after filtering
        if filtered_items:
            filtered_transactions[transaction_id] = filtered_items

    return filtered_transactions


def sort_transaction_items_by_total_order(
    transactions: dict, sorted_secondary_a: list
) -> dict:
    """
    Sort the items in each transaction based on a given total order.

    Parameters:
    transactions (dict): A dictionary of transactions, where each transaction is a dictionary of items and their quantities.
    sorted_secondary_a (list): A list of items sorted according to a specific order.

    Returns:
    dict: A dictionary of sorted transactions, where each transaction is a dictionary of items and their quantities.
          Only transactions containing items from sorted_secondary_a are included in the result.
          The items in each transaction are sorted based on the total order.
    """
    # Create a dictionary to store the order of each item based on sorted_secondary_a
    item_order = {item: index for index, item in enumerate(sorted_secondary_a)}

    # Sort the items in each transaction based on the defined order
    sorted_transactions = {}
    for transaction_id, items in transactions.items():
        sorted_items = dict(
            sorted(
                items.items(), key=lambda item: item_order.get(item[0], float("inf"))
            )
        )
        sorted_transactions[transaction_id] = sorted_items

    return sorted_transactions


def transform_to_utilities(transactions: dict, profits: dict) -> list:
    """
    Convert the transaction database to utilities (profit * quantity) for each item.

    Parameters:
    transactions (dict): A dictionary of transactions, where each transaction is a dictionary of items and their quantities.
    profits (dict): A dictionary of item profits, where the keys are item IDs and the values are the corresponding profits.

    Returns:
    list: A list of transformed transactions, where each transaction is a dictionary of items and their utilities.
    """
    transformed_db = []

    for transactionID, items in transactions.items():
        transformed_transaction = {}
        for itemID, quantity in items.items():
            # Calculate the utility for each item in the transaction
            profit = profits.get(itemID, 0)
            utility = profit * quantity
            transformed_transaction[itemID] = utility
        transformed_db.append(transformed_transaction)

    return transformed_db


def calculate_utility_of_itemset(transaction: dict, itemset: list) -> float:
    """
    Calculate the total utility of an itemset in a given transaction.

    Parameters:
    transaction (dict): A dictionary representing a transaction, where keys are item IDs and values are their quantities/profits.
    itemset (list): A list of item IDs representing the itemset for which the utility needs to be calculated.

    Returns:
    float: The total utility of the itemset in the given transaction. If any item in the itemset is not found in the transaction, the function returns 0.
    """
    total_utility = 0

    if itemset == []:
        return 0

    for item in itemset:
        # Check if the item is in the transaction and calculate its utility
        if item in transaction:
            utility = transaction[item]
            total_utility += utility  # Add the utility of the item to the total
        else:
            return 0
    return total_utility


def calculate_utility_all(database: list, itemset: list) -> float:
    """
    Calculate the total utility of an itemset in a given transaction database.

    Parameters:
    database (list): A list of transactions, where each transaction is a dictionary of items and their utilities.
    itemset (list): A list of item IDs representing the itemset for which the utility needs to be calculated.

    Returns:
    float: The total utility of the itemset in the given transaction database.
    """
    utility = 0
    for transaction in database:
        utility += calculate_utility_of_itemset(transaction, itemset)
    return utility


def calculate_psu(sorted_database: list, X: list, target_item: str) -> float:
    """
    Calculate the PSU value for a set X and a target item in a sorted transaction database.

    Parameters:
    sorted_database (list): A list of transactions, where each transaction is a dictionary of items and their utilities.
    X (list): A list of item IDs representing the itemset for which the utility needs to be calculated.
    target_item (str): The ID of the target item for which the PSU value needs to be calculated.

    Returns:
    float: The PSU value for the given set X and target item in the sorted transaction database.
    """
    psu = 0

    # Iterate through each transaction in the database
    for transaction in sorted_database:
        if target_item in transaction:  # If the transaction contains the target_item
            start_counting = False
            item_utility = 0
            rru = 0

            # Iterate through all items in the transaction
            for i in transaction:

                # Calculate the combined utility of the itemset X in this transaction
                if (all(item in transaction for item in X)) or X == []:
                    # Start calculating utility after encountering the target_item
                    if i == target_item:
                        start_counting = True
                        item_utility = transaction[i] + calculate_utility_of_itemset(
                            transaction, X
                        )
                    elif start_counting and transaction[i] > 0:
                        rru += transaction[i]  # Remaining utility after the target_item

            # Add the utility of the psu
            psu += item_utility + rru

    return psu


def calculate_plu(
    sorted_database: list, X: list, target_item: str, ptwu: dict
) -> float:
    """
    Calculate the PLU value for a set X and a target item in a sorted transaction database.

    Parameters:
    sorted_database (list): A list of transactions, where each transaction is a dictionary of items and their utilities.
    X (list): A list of item IDs representing the itemset for which the utility needs to be calculated.
    target_item (str): The ID of the target item for which the PLU value needs to be calculated.
    ptwu (dict): A dictionary representing the Ptwu values for each item in the database.

    Returns:
    float: The PLU value for the given set X and target item in the sorted transaction database.
    """
    plu = 0
    if X == []:
        return ptwu[target_item]
    for transaction in sorted_database:
        if all(item in transaction for item in X):
            if target_item in transaction:
                start_counting = False
                item_utility = 0
                rru = 0
                utility_of_X = calculate_utility_of_itemset(transaction, X)
                for i in transaction:
                    # Start counting utilities after reaching the target item
                    if i == X[-1]:
                        start_counting = True
                    elif start_counting and transaction[i] > 0:
                        rru += transaction[
                            i
                        ]  # Remaining positive utility after target item
                plu += (
                    utility_of_X + rru
                )  # Sum of target item utility and remaining utilities
    return plu


def transaction_projection(transaction: dict, itemset: list) -> dict:
    """
    Project a transaction based on a given itemset.

    Parameters:
    transaction (dict): A dictionary representing a transaction, where keys are item IDs and values are their quantities.
    itemset (list): A list of item IDs representing the itemset for projection.
    Returns:
    dict: A dictionary representing the projected transaction, containing only the items and quantities that are after the last item in the itemset in the original transaction.
    """
    projected_transaction = {}
    # Convert itemset to a set for faster lookup
    itemset_items = set(itemset)
    # Check if all items in the itemset are present in the transaction
    if itemset_items.issubset(transaction.keys()):
        # Get the list of items in the transaction
        items = list(transaction.keys())
        quantities = list(transaction.values())

        # Find the index of the last item in the itemset in the transaction
        last_index = -1
        for i, item in enumerate(items):
            if item in itemset_items:
                last_index = i

        # If the last item in the itemset is found, get the items and quantities after it
        if last_index != -1:
            # The items and quantities after the last item in the itemset
            for i in range(last_index + 1, len(items)):
                projected_transaction[items[i]] = quantities[
                    i
                ]  # Save to projected_transaction

    return projected_transaction


def database_projection(dataset: list, itemset: list) -> list:
    """
    Projects a transaction database based on a given itemset.

    Parameters:
    dataset (list): A list of transactions, where each transaction is a dictionary.
    itemset (list): A list of item IDs representing the itemset for projection.

    Returns:
    list: A list of projected transactions, containing only the items and quantities that are after the last item in the itemset in the original transaction.
    """
    projected_dataset = []

    for transaction in dataset:
        # Call transaction_projection function for each transaction in dataset
        projected_transaction = transaction_projection(transaction, itemset)
        # If the projected transaction is not empty, add it to the result dataset
        if projected_transaction:
            projected_dataset.append(projected_transaction)

    return projected_dataset


def merge_transactions(projected_dataset: list) -> list:
    """
    Merges transactions in a projected dataset by combining items with the same itemset.

    Parameters:
    projected_dataset (list): A list of dictionaries, where each dictionary represents a transaction.

    Returns:
    list: A list of merged transactions, where each transaction is a dictionary.
    """
    merged_dataset = []

    # Iterate through each transaction in projected_dataset
    for transaction in projected_dataset:
        # Convert transaction to tuple (sorted) for easy comparison
        sorted_items = tuple(sorted(transaction.keys()))
        existing_transaction = None

        # Check if there is any transaction already with the same itemset
        for merged_transaction in merged_dataset:
            if tuple(sorted(merged_transaction.keys())) == sorted_items:
                existing_transaction = merged_transaction
                break

        if existing_transaction:
            # If a similar transaction is found, add the quantities of items to the existing transaction
            for item in sorted_items:
                existing_transaction[item] += transaction[item]
        else:
            # If no similar transaction is found, add the transaction to the merged_dataset
            merged_dataset.append(transaction)

    return merged_dataset


def search(
    alpha: list,
    prefix_utility: float,
    eta: set,
    projected_dataset: list,
    primary_items: list,
    secondary_items: list,
    minUtil: float,
    ptwu: dict,
    k: int,
) -> None:
    """
    Performs a depth-first search to find high-utility itemsets (HUIs) in a projected transaction database.

    Parameters:
    alpha (list): The current itemset being extended.
    prefix_utility (float): The utility of the current itemset.
    eta (set): The set of promising negative items.
    projected_dataset (list): The projected transaction database.
    primary_items (list): The primary items to be considered for extension.
    secondary_items (list): The secondary items to be considered for extension.
    minUtil (float): The minimum utility threshold.
    ptwu (dict): The positive transaction-weighted utility (Ptwu) of each item.
    k (int): The required number of HUIs.

    Returns:
    None. The function updates a global variable `TKHQ` with the top-k HUIs.
    """
    # Iterate through each item z in the primary_items of alpha
    global TKHQ  # Use the global variable TKHQ
    for z in primary_items:
        # Initialize beta as the extension of alpha with z
        beta = alpha + [z]
        # Calculate the utility of beta and create the beta dataset - D
        beta_utility = calculate_utility_all(projected_dataset, beta)

        # Check if the utility of beta >= minUtil
        if beta_utility >= minUtil:
            # Add beta to the priority queue TKHQ
            TKHQ.append((beta, beta_utility))
            TKHQ = sorted(
                TKHQ, key=lambda x: x[1], reverse=True
            )  # Sort TKHQ by descending utility
            # If TKHQ has k elements, increase minUtil to the utility of the highest element
            if len(TKHQ) > k:
                TKHQ.pop()  # Remove the element with the lowest utility
            if len(TKHQ) == k:
                minUtil = TKHQ[-1][
                    1
                ]  # minUtil is the utility of the smallest element in TKHQ
        w = None
        # Find the next item `w` in `secondary_items` (no binary search used)
        zindex = secondary_items.index(z)
        windex = zindex + 1
        if windex > len(secondary_items):
            continue
        # If the utility of beta < minUtil and w is a negative item
        if beta_utility < minUtil and w in eta:
            continue  # Skip and continue to the next iteration
        remaining_secondary_items = secondary_items[windex::]
        # Define primary(beta) and secondary(beta)
        # Calculate PSU and PLU of beta and remaining items in secondary_items
        psu_beta_w = {
            w: calculate_psu(projected_dataset, beta, w)
            for w in remaining_secondary_items
        }
        plu_beta_w = {
            w: calculate_plu(projected_dataset, beta, w, ptwu)
            for w in remaining_secondary_items
        }
        # Define primary(beta) and secondary(beta)
        primary_beta = [
            w for w in remaining_secondary_items if psu_beta_w[w] >= minUtil
        ]
        secondary_beta = [
            w for w in remaining_secondary_items if plu_beta_w[w] >= minUtil
        ]
        # Recursively call the search function to extend beta with items in primary_items using DFS
        search(
            beta,
            beta_utility,
            eta,
            projected_dataset,
            primary_beta,
            secondary_beta,
            minUtil,
            ptwu,
            k,
        )


def TKN_algorithm(D: list, k: int) -> None:
    """
    Performs the Top-k High-Utility Itemset (TKHI) algorithm.

    Parameters:
    D (list): A transaction dataset.
    k (int): The required number of HUIs.

    Returns:
    None. Prints the top-k HUIs.
    """
    Alpha = []
    minUtil = 0

    # Step 1: Calculate PRIU for each item
    priu_dict = calculate_priu(D, profits)

    # Step 2: Let g be the set of promising negative items in D
    # Define threshold to classify promising negative items
    g = {item for item, utility in priu_dict.items() if utility < 0}

    ptwu = calculate_ptwu(transactions, profits)

    # Step 5: Calculate PRIU for all items with positive PRIU and store them in PRIU_list
    priu_dict = calculate_priu(D, profits)
    g = {
        item for item, utility in priu_dict.items() if utility < 0
    }  # Set of items with negative PRIU

    # Step 6: Apply PRIU strategy to increase minUtil to the value of the kth highest PRIU
    minUtil = priu_strategy(priu_dict, k)

    # Step 9: Calculate secondary(a) for items with Ptwu >= minUtil
    secondary_a = {item for item, ptwu_value in ptwu.items() if ptwu_value >= minUtil}

    # Step 9: Sort secondary(a) in total order
    sorted_secondary_a = sort_secondary_a_by_total_order(secondary_a, ptwu, profits)

    # Step 10: Scan D and remove items not in secondary(a)
    filtered_transactions = filter_transactions_by_secondary_a(D, sorted_secondary_a)

    # Step 11: Sort transactions
    sorted_transactions = sort_transaction_items_by_total_order(
        filtered_transactions, sorted_secondary_a
    )

    # Transform db to profit-based db
    db_transform = transform_to_utilities(sorted_transactions, profits)

    # Step 12: Calculate PSU for each item in secondary(a)
    PSU_table = {}
    for item in sorted_secondary_a:
        PSU_table[item] = calculate_psu(db_transform, Alpha, item)

    # Step 13: Build LIUS
    LIUS = LIUS_Build_contiguous(db_transform, priu_dict)

    # Step 14: Apply PLIU-E and PLIU-LB strategies
    PLIU_E, PIQU_LIU = PLIU_E_strategy(LIUS, minUtil, k)
    PLIU_LB = PLIU_LB_strategy(LIUS, priu_dict, PIQU_LIU, k, PLIU_E)

    # Step 16: Identify primary items based on PSU
    primary_items = [item for item, psu in PSU_table.items() if psu >= PLIU_LB]

    # Perform depth-first search to find top-k HUIs
    topkHUIs = search(
        Alpha, 0, g, db_transform, primary_items, sorted_secondary_a, PLIU_LB, ptwu, k
    )

    print("Result:", TKHQ[0:k])


k = 4
result = TKN_algorithm(transactions, k)
