import itertools


def flattened_ranges(range_lengths):
    return list(itertools.chain.from_iterable(range(n) for n in range_lengths))


def take(n, gen):
    """Take the first n items from the generator gen. Return a list of that many items
    and the resulting state of the generator."""
    lst = list(itertools.islice(gen, n))
    return lst, gen


def flatten_list(seq):
    """Flatten a list-of-lists into a single list. This does only one level of flattening."""
    return sum(seq, [])


def repeated_indices(repetition_counts):
    """Given a sequence of repetition counts, the index values 0..len(repetition_counts)-1,
    each value repeated as many times as the corresponding entry in repetition_counts.

    Example:
      repeated_indices([2,0,3]) -> [0, 0, 2, 2, 2]
    """
    return flatten_list(
        list(itertools.repeat(i, rc)) for (i, rc) in enumerate(repetition_counts)
    )


def make_outer_indices(outer_count, inner_count):
    """Given a sequence of counts 'outer_count' of an *outer* object type,
    and a sequence of counts 'inner_count' for the number of *inner* objects in each
    *outer* object, return a sequence of the indices identifying, for each inner object,
    the index of the outer object to which it corresponds.

    Example:
      make_outer_indices([1, 2], [2, 1, 2]) -> [0, 0, 0, 1, 1]
    """
    ic = (x for x in inner_count)
    temp = (take(oc, ic)[0] for oc in outer_count)
    return flatten_list(repeated_indices(x) for x in temp)
