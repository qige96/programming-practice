

def coverage(item_set, txs):
    '''
    count the number of instances covered by the rule

    Parameters
    ----------
    item_set: list
        item set with any number of items
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    count: int
        coverage of a rule given a set of transaactions

    Examples
    --------
        from Lecure 6 Slide 12
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> set1 = ('o',); coverage(set1, txs)
        4
        >>> set2 = ('o','s'); coverage(set2, txs)
        2
    '''
    count = 0
    for tx in txs:
        if set(tx).issuperset(item_set):
            count += 1
    return count

def support(item_set, txs):
    '''
    calculate the proportion of instances covered by the rule
    
    Parameters
    ----------
    item_set: list
        item set with any number of items
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    proportion: float
        proportion of a rule gain supported given a set of transaactions
    '''
    return coverage(item_set, txs) / len(txs)

def confidence(rule, txs):
    '''
    calculate the proportion of instances that the rule predicts 
    correctly over all instances. Also called accuracy.

    Parameters
    ----------
    rule: tuple
        tuple of tuples, where tup[0] is antecedent and tup[1] is consequent
    txs: list
        list of transactions that is a list of items

    Returns
    -------
    confid: float
        confidence of the given rule

    Examples
    --------
        from Lecure 6 Slide 16
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> rule1 = (('o',), ('s',)); confidence(rule1, txs)
        0.5
        >>> rule2 = (('s',), ('o',)); confidence(rule2, txs)
        0.6666666666666666
        >>> rule3 = (('o',), ('d',)); confidence(rule3, txs)
        0.5
        >>> rule4 = (('d',), ('o',)); confidence(rule4, txs)
        1.0
    '''
    return coverage(rule[0]+rule[1], txs) / coverage(rule[0], txs)

def filter_item_sets_with_coverage(item_sets, txs, lower_bound):
    '''
    filter out item set with given minimum coverage

    Parameters
    ----------
    item_sets: list
        list of item sets
    txs: list
        list of transactions
    lower_bound: int
        minimun coverage that an item set should meet

    Returns
    -------
    new_set : list
        a new list of item sets in which all item sets meet minimun coverage
    '''
    new_set = []
    for i in item_sets:
        if coverage(i, txs) >= lower_bound:
            new_set.append(i)
    return new_set

def get_items(item_sets):
    '''extreact items from item sets or transactions'''
    items = []
    for item_set in item_sets:
        for item in item_set:
            if item not in items:
                items.append(item)
    return items

def one_item_sets(txs, min_coverage):
    '''
    construct one-item set from a batch of transactions
    
    Parameters
    ----------
    txs: list
        list of transactions that is a list of items
    min_coverage: int
        minumun coverage the item set should meet

    Returns
    -------
    one_set: list
        list of one-item set

    Examples
    --------
        from Lecure 6 Slide 12
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> one_item_sets(txs, 2)
        [('o',), ('s',), ('w',), ('d',)]
        >>> one_item_sets(txs, 1)
        [('o',), ('s',), ('m',), ('w',), ('d',)]
    '''
    items = get_items(txs)
    one_sets = [(i,) for i in items]
    return filter_item_sets_with_coverage(one_sets, txs, min_coverage)

def two_item_sets(one_sets, txs, min_coverage):
    '''
    construct two-item set from one-item set
    
    Parameters
    ----------
    one_sets: list
        one-item sets
    txs: list
        list of transactions that is a list of items
    min_coverage: int
        minumun coverage the item set should meet

    Returns
    -------
    two_sets: list
        list of two-item sets

    Examples
    --------
        from Lecure 6 Slide 13
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> one_sets = [('o',), ('s',), ('w',), ('d',)]
        >>> two_item_sets(one_sets, txs, 2) 
        [('o', 's'), ('o', 'd')]
        >>> two_item_sets(one_sets, txs, 1)
        [('o', 's'), ('o', 'w'), ('o', 'd'), ('s', 'w'), ('s', 'd')]
    '''
    two_sets = []
    items = get_items(one_sets)
    from itertools import combinations_with_replacement
    for tup in combinations_with_replacement(items, 2):
        if len(set(tup)) == 2:
            two_sets.append(tup)
    return filter_item_sets_with_coverage(two_sets, txs, min_coverage)

def three_item_sets(two_sets, txs, min_coverage):
    '''
    construct three-item set from one-item set
    
    Parameters
    ----------
    two_sets: list
        two-item sets
    txs: list
        list of transactions that is a list of items
    min_coverage: int
        minumun coverage the item set should meet

    Returns
    -------
    three_sets: list
        list of three-item sets

    Examples
    --------
        from Lecure 6 Slide 13
        >>> txs = [['o','s'], ['m','o','w'], ['o','d'], ['o','d','s'], ['w','s']]
        >>> two_sets = [('o','s'), ('o','d')]
        >>> three_item_sets(two_sets, txs, 2)
        []
        >>> three_item_sets(two_sets, txs, 1)
        [('o', 's', 'd')]
    '''
    three_sets = []
    items = get_items(two_sets)
    from itertools import combinations_with_replacement
    for tup in combinations_with_replacement(items, 3):
        if len(set(tup)) == 3:
            three_sets.append(tup)
    return filter_item_sets_with_coverage(three_sets, txs, min_coverage)


