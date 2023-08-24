# -*- coding: utf-8 -*-
# Natural Language Toolkit: Distance Metrics
#
# Copyright (C) 2001-2020 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
#         Steven Bird <stevenbird1@gmail.com>
#         Tom Lippincott <tom@cs.columbia.edu>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT
#

"""
Distance Metrics.

Compute the distance between two items (usually strings).
As metrics, they must satisfy the following three requirements:

1. d(a, a) = 0
2. d(a, b) >= 0
3. d(a, c) <= d(a, b) + d(b, c)
"""

import warnings
import operator

# Import relevant libraries and dependencies
from typing import List, Union, Dict, Tuple
import numpy as np

"""
add string2string module code, src = https://github.com/stanfordnlp/string2string

MIT License

Copyright (c) 2023 Mirac Suzgun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Parent class for all the string algorithms implemented in this module
class StringAlgs:
    """
        This class is the parent class for all the string algorithms implemented in this module.
    """
    # Initialize the class
    def __init__(self,
                match_weight: float = 0.0,
        ) -> None:
        # Set the match weight
        self.match_weight = match_weight

# Levenshtein edit distance class
class LevenshteinEditDistance(StringAlgs):
    def __init__(self,
                match_weight: float = 0.0,
                insert_weight: float = 1.0,
                delete_weight: float = 1.0,
                substitute_weight: float = 1.0,
    ) -> None:
        r"""
        This class initializes the Levenshtein edit distance algorithm. Levenshtein edit distance represents the minimum number of edit distance operations (insertion, deletion, and substitution) required to convert one string to another.
            
        The Levenshtein edit distance (with unit cost for each edit distance operation) is given by the following recurrence relation: 

        .. math::
            :nowrap:

            \begin{align}
            d[i, j] := \min( & d[i-1, j-1] + \texttt{mismatch}(i, j),  \\
                                & d[i-1, j] + 1,  \\
                                & d[i, j-1] + 1),
            \end{align}

        where :math:`\texttt{mismatch}(i, j)` is 1 if the i-th element in str1 is not equal to the j-th element in str2, and 0 otherwise.

        Arguments:
            match_weight (float): The weight of a match (default: 0.0).
            insert_weight (float): The weight of an insertion (default: 1.0).
            delete_weight (float): The weight of a deletion (default: 1.0).
            substitute_weight (float): The weight of a substitution (default: 1.0).

        Raises:
            AssertionError: If any of the weights are negative.
        """
        # Set the match weight
        super().__init__(match_weight=match_weight)

        # Set the insert, delete, and substite weights
        self.insert_weight = insert_weight
        self.delete_weight = delete_weight
        self.substitute_weight = substitute_weight

        # Assert that all the weights are non-negative
        assert min(match_weight, insert_weight, delete_weight, substitute_weight) >= 0.0


    
    # Compute the Levenshtein edit distance between two strings using recursion
    def compute_recursive(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> float:
        r"""
        This function computes the Levenshtein edit distance between two strings (or lists of strings) using recursion.

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The Levenshtein edit distance between the two strings.

        .. note::
            * The solution presented here utilizes recursion to compute the Levenshtein edit distance between two strings. It has an exponential time complexity and is not recommended for pairs of strings with a large length.
            * The time complexity of this function is :math:`O(3^{m+n})`, where :math:`m` and :math:`n` are the lengths of the two strings.
        """
        # Base case
        if len(str1) == 0:
            return len(str2) * self.insert_weight
        elif len(str2) == 0:
            return len(str1) * self.delete_weight

        # Compute the mismatch
        mismatch = 0.0 if str1[-1] == str2[-1] else self.substitute_weight

        # Compute the Levenshtein edit distance
        return min(
            self.compute_recursive(str1[:-1], str2[:-1]) + mismatch,
            self.compute_recursive(str1[:-1], str2) + self.delete_weight,
            self.compute_recursive(str1, str2[:-1]) + self.insert_weight,
        )

    

    # Compute the Levenshtein edit distance between two strings using memoization
    def compute_recursive_memoization(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
    ) -> float:
        r"""
        This function computes the Levenshtein edit distance between two strings (or lists of strings) using memoization.

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The Levenshtein edit distance between the two strings.

        .. note::
            * The solution presented here utilizes memoization to compute the Levenshtein edit distance between two strings. 
            * The time complexity of this function is :math:`\mathcal{O}(m n)`, where :math:`m` and :math:`n` are the lengths of the two strings.
        """
        # Initialize the memoization dictionary
        memoization = {}

        # Compute the Levenshtein edit distance
        return self.compute_memoization_helper(str1, str2, memoization)

    

    # Compute the Levenshtein edit distance between two strings using memoization (helper function)
    def compute_memoization_helper(self,
        str1: Union[str, List[str]],
        str2: Union[str, List[str]],
        memoization: Dict[Tuple[str, str], float],
    ) -> float:
        r"""
        This is a helper function that computes the Levenshtein edit distance between two strings (or lists of strings) using memoization.

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).
            memoization (dict): The memoization dictionary.

        Returns:
            The Levenshtein edit distance between the two strings.

        .. note::
            * The solution presented here utilizes memoization to compute the Levenshtein edit distance between two strings.
            * One can also use the :func:`functools.lru_cache` (@lru_cache()) decorator to memoize the function calls. However, for the sake of educational purposes, we have implemented memoization using a dictionary.
            * The time complexity of this function is quadratic, that is :math:`\mathcal{O}(nm)`, where m and n are the lengths of the two strings.
        """
        # Base case
        if len(str1) == 0:
            return len(str2) * self.insert_weight
        elif len(str2) == 0:
            return len(str1) * self.delete_weight

        # Check if the Levenshtein edit distance has already been computed
        if (str1, str2) in memoization:
            return memoization[(str1, str2)]

        # Compute the mismatch
        mismatch = 0.0 if str1[-1] == str2[-1] else self.substitute_weight

        # Compute the Levenshtein edit distance
        memoization[(str1, str2)] = min(
            self.compute_memoization_helper(str1[:-1], str2[:-1], memoization) + mismatch,
            self.compute_memoization_helper(str1[:-1], str2, memoization) + self.delete_weight,
            self.compute_memoization_helper(str1, str2[:-1], memoization) + self.insert_weight,
        )

        # Return the Levenshtein edit distance
        return memoization[(str1, str2)]



    # Compute the Levenshtein edit distance between two strings using dynamic programming
    def compute_dynamic_programming(self,
        str1: Union[str, List[str]], 
        str2: Union[str, List[str]],
    ) -> float:
        r"""
        This function computes the Levenshtein edit distance between two strings (or lists of strings) using dynamic programming (Wagner-Fischer algorithm).

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The Levenshtein edit distance between the two strings.

        .. note::
            * The solution presented here utilizes dynamic programming principles to compute the Levenshtein edit distance between two strings. 
            * This solution is also known as the Wagner-Fischer algorithm. [WF1974]_
            * The time complexity of this dynamic-programming-based solution is :math:`\mathcal{O}(nm)`, and the space complexity is :math:`\mathcal{O}(nm)`, where n and m are the lengths of the two strings, respectively.
            * However, by using only two rows of the distance matrix at a time, the space complexity of the dynamic programming solution can be reduced to :math:`\mathcal{O}(min(n, m))`.
            * The time complexity cannot be made strongly subquadratic time unless SETH is false. [BI2015]_
            * Finally, we note that this solution can be extended to cases where each edit distance operation has a non-unit cost.

            .. [WF1974] Wagner, R.A. and Fischer, M.J., 1974. The string-to-string correction problem. Journal of the ACM (JACM), 21(1), pp.168-173.
            .. [BI2015] Backurs, A. and Indyk, P., 2015, June. Edit distance cannot be computed in strongly subquadratic time (unless SETH is false). In Proceedings of the forty-seventh annual ACM symposium on Theory of computing (pp. 51-58).
        """
        # Lengths of strings str1 and str2, respectively.
        n = len(str1)
        m = len(str2)

        # Initialize the distance matrix.
        dist = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            dist[i, 0] = self.delete_weight * i
        for j in range(1, m + 1):
            dist[0, j] = self.insert_weight * j

        # Dynamic programming step, where each operation has a unit cost:
        # d[i, j] := min(d[i-1, j-1] + mismatch(i, j), d[i-1, j] + 1, d[i, j-1] + 1),
        # where mismatch(i, j) is 1 if str1[i] != str2[j] and 0 otherwise.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Compute the minimum edit distance between str1[:i] and str2[:j].
                dist[i, j] = min(
                    dist[i-1, j-1] + (self.substitute_weight if str1[i-1] != str2[j-1] else self.match_weight),
                    dist[i-1, j] + self.delete_weight, 
                    dist[i, j-1] + self.insert_weight,
                )

        # Return the Levenshtein edit distance between str1 and str2.
        return dist[n, m]



    # Compute the Levenshtein edit distance between two strings
    def compute(self,
        str1: Union[str, List[str]], 
        str2: Union[str, List[str]],
        method: str = "dynamic-programming",
    ) -> float:
        r"""
        This function computes the Levenshtein edit distance between two strings (or lists of strings), using the method specified by the user. 

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).
            method (str): The method to use to compute the Levenshtein edit distance (default: "dynamic-programming").

        Returns:
            The Levenshtein edit distance between the two strings.

        .. note::
            * The method can be one of the following:
                * "recursive": This method computes the Levenshtein edit distance using recursion.
                * "recursive-memoization": This method computes the Levenshtein edit distance using recursion with memoization.
                * "dynamic-programming": This method computes the Levenshtein edit distance using dynamic programming (Wagner-Fischer algorithm).
            * By default, the method is "dynamic-programming".
            
        """
        # If the method is dynamic programming, then compute the Levenshtein edit distance using dynamic programming
        if method == "recursive":
            return self.compute_recursive(str1, str2)
        elif method == "recursive-memoization":
            return self.compute_recursive_memoization(str1, str2)
        return self.compute_dynamic_programming(str1, str2)

# Hamming (edit) distance class
class HammingDistance(StringAlgs):
    def __init__(self, 
                match_weight: float = 0.0,
                substitute_weight: float = 1.0,
        ) -> None:
        r"""
        This function initializes the class variables of the Hamming distance. 
        
        The Hamming distance is the number of positions at which the corresponding symbols are different. [H1950]_

        Arguments:
            match_weight (float): The weight of a match (default: 0.0).
            substitute_weight (float): The weight of a substitution (default: 1.0).

        Raises:
            AssertionError: If the substite weight is negative.

        .. note::
            * The Hamming distance has a time complexity of :math:`\mathcal{O}(n)`, where :math: `n` the length of the two strings.

        .. [H1950] Hamming, R.W., 1968. Error detecting and error correcting codes. Bell System Technical Journal, 29(2), pp.147-160.
        """
        # Set the match weight
        super().__init__(match_weight=match_weight)

        # Set the substite weight
        self.substitute_weight = substitute_weight

        # Assert that the substite weight is non-negative
        assert substitute_weight >= 0.0



    # Compute the Hamming distance between two strings
    def compute(self,
        str1: Union[str, List[str]], 
        str2: Union[str, List[str]],
    ) -> float:
        """
        This function computes the Hamming distance between two strings (or lists of strings).

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The Hamming distance between the two strings.

        Raises:
            ValueError: If the two strings (or lists of strings) have different lengths.
        """

        # Lengths of strings str1 and str2, respectively.
        n = len(str1)
        m = len(str2)

        # Assert that the two strings have the same length
        if n != m:
            raise ValueError("The two strings (or lists of strings) must have the same length.")

        # Compute the Hamming edit distance between str1 and str2.
        return sum(
            self.substitute_weight if str1[i] != str2[i] else self.match_weight
            for i in range(n)
        )


# Damerau-Levenshtein edit distance class
class DamerauLevenshteinDistance(LevenshteinEditDistance):
    def __init__(self, 
                match_weight: float = 0.0,
                insert_weight: float = 1.0,
                delete_weight: float = 1.0,
                substitute_weight: float = 1.0,
                adjacent_transpose_weight: float = 1.0,
        ) -> None:
        r"""
        This function initializes the class variables of the Damerau-Levenshtein distance.
         
        The Damerau-Levenshtein distance is the minimum number of insertions, deletions, substitutions, and transpositions required to transform one string into the other. [D1964]_

        Arguments:
            match_weight (float): The weight of a match (default: 0.0).
            insert_weight (float): The weight of an insertion (default: 1.0).
            delete_weight (float): The weight of a deletion (default: 1.0).
            substitute_weight (float): The weight of a substitution (default: 1.0).
            adjacent_transpose_weight (float): The weight of an adjacent transposition (default: 1.0).

        Raises:
            AssertionError: If the insert, delete, substite, or adjacent transpose weights are negative.

        .. [D1964] Damerau, F.J., 1964. A technique for computer detection and correction of spelling errors. Communications of the ACM, 7(3), pp.171-176.
        """
        # Set the weights of the distance operations
        super().__init__(
            match_weight=match_weight,
            insert_weight=insert_weight,
            delete_weight=delete_weight,
            substitute_weight=substitute_weight,
        )

        # Set the adjacent transpose weight
        self.adjacent_transpose_weight = adjacent_transpose_weight

        # Assert that the adjacent transpose weight is non-negative
        assert adjacent_transpose_weight >= 0.0



    # Compute the Damerau-Levenshtein edit distance between two strings
    def compute(self,
        str1: Union[str, List[str]], 
        str2: Union[str, List[str]],
    ) -> float:
        """
        This function computes the Damerau-Levenshtein edit distance between two strings (or lists of strings).

        Arguments:
            str1 (str or list of str): The first string (or list of strings).
            str2 (str or list of str): The second string (or list of strings).

        Returns:
            The Damerau-Levenshtein distance between the two strings.

        .. note::
            * The Damerau-Levenshtein distance is a variant of the Levenshtein distance that allows for adjacent transpositions.
            * The dynamic programming solution to the Damerau-Levenshtein distance has a time complexity of :math:`\mathcal{O}(nm)`, where n and m are the lengths of the two strings.
        """

        # Lengths of strings str1 and str2, respectively.
        n = len(str1)
        m = len(str2)

        # Initialize the distance matrix.
        dist = np.zeros((n + 1, m + 1))
        for i in range(1, n + 1):
            dist[i, 0] = self.delete_weight * i
        for j in range(1, m + 1):
            dist[0, j] = self.insert_weight * j

        # Dynamic programming solution to the Damerau-Levenshtein edit distance is very similar to that of the Levenshtein edit distance.
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dist[i, j] = min(
                    dist[i-1, j-1] + (self.substitute_weight if str1[i-1] != str2[j-1] else self.match_weight),
                    dist[i-1, j] + self.delete_weight, 
                    dist[i, j-1] + self.insert_weight,
                )
                # This is the only difference between the Damerau-Levenshtein edit distance and the Levenshtein edit distance.
                if i > 1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
                    dist[i, j] = min(dist[i, j], dist[i-2, j-2] + self.adjacent_transpose_weight)

        # Return the Damerau-Levenshtein edit distance between str1 and str2.
        return dist[n, m]

"""
string2string code ends here
"""

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    """
    Calculate the Levenshtein edit-distance between two strings.
    The edit distance is the number of characters that need to be
    substituted, inserted, or deleted, to transform s1 into s2.  For
    example, transforming "rain" to "shine" requires three steps,
    consisting of two substitutions and one insertion:
    "rain" -> "sain" -> "shin" -> "shine".  These operations could have
    been done in other orders, but at least three steps are needed.

    Allows specifying the cost of substitution edits (e.g., "a" -> "b"),
    because sometimes it makes sense to assign greater penalties to
    substitutions.

    This also optionally allows transposition edits (e.g., "ab" -> "ba"),
    though this is disabled by default.

    :param s1, s2: The strings to be analysed
    :param transpositions: Whether to allow transposition edits
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :type transpositions: bool
    :rtype int
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )
    return lev[len1][len2]


def _edit_dist_backtrace(lev):
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment = [(i, j)]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j))
    return list(reversed(alignment))


def edit_distance_align(s1, s2, substitution_cost=1):
    """
    Calculate the minimum Levenshtein edit-distance based alignment
    mapping between two strings. The alignment finds the mapping
    from string s1 to s2 that minimizes the edit distance cost.
    For example, mapping "rain" to "shine" would involve 2
    substitutions, 2 matches and an insertion resulting in
    the following mapping:
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (4, 5)]
    NB: (0, 0) is the start state without any letters associated
    See more: https://web.stanford.edu/class/cs124/lec/med.pdf

    In case of multiple valid minimum-distance alignments, the
    backtrace has the following operation precedence:
    1. Skip s1 character
    2. Skip s2 character
    3. Substitute s1 and s2 characters
    The backtrace is carried out in reverse string order.

    This function does not support transposition.

    :param s1, s2: The strings to be aligned
    :type s1: str
    :type s2: str
    :type substitution_cost: int
    :rtype List[Tuple(int, int)]
    """
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=False,
            )

    # backtrace to find alignment
    alignment = _edit_dist_backtrace(lev)
    return alignment


def binary_distance(label1, label2):
    """Simple equality test.

    0.0 if the labels are identical, 1.0 if they are different.

    >>> from nltk.metrics import binary_distance
    >>> binary_distance(1,1)
    0.0

    >>> binary_distance(1,3)
    1.0
    """

    return 0.0 if label1 == label2 else 1.0


def jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity.

    """
    return (len(label1.union(label2)) - len(label1.intersection(label2))) / len(
        label1.union(label2)
    )


def masi_distance(label1, label2):
    """Distance metric that takes into account partial agreement when multiple
    labels are assigned.

    >>> from nltk.metrics import masi_distance
    >>> masi_distance(set([1, 2]), set([1, 2, 3, 4]))
    0.665

    Passonneau 2006, Measuring Agreement on Set-Valued Items (MASI)
    for Semantic and Pragmatic Annotation.
    """

    len_intersection = len(label1.intersection(label2))
    len_union = len(label1.union(label2))
    len_label1 = len(label1)
    len_label2 = len(label2)
    if len_label1 == len_label2 and len_label1 == len_intersection:
        m = 1
    elif len_intersection == min(len_label1, len_label2):
        m = 0.67
    elif len_intersection > 0:
        m = 0.33
    else:
        m = 0

    return 1 - len_intersection / len_union * m


def interval_distance(label1, label2):
    """Krippendorff's interval distance metric

    >>> from nltk.metrics import interval_distance
    >>> interval_distance(1,10)
    81

    Krippendorff 1980, Content Analysis: An Introduction to its Methodology
    """

    try:
        return pow(label1 - label2, 2)
    #        return pow(list(label1)[0]-list(label2)[0],2)
    except:
        print("non-numeric labels not supported with interval distance")


def presence(label):
    """Higher-order function to test presence of a given label
    """

    return lambda x, y: 1.0 * ((label in x) == (label in y))


def fractional_presence(label):
    return (
        lambda x, y: abs(((1.0 / len(x)) - (1.0 / len(y))))
        * (label in x and label in y)
        or 0.0 * (label not in x and label not in y)
        or abs((1.0 / len(x))) * (label in x and label not in y)
        or ((1.0 / len(y))) * (label not in x and label in y)
    )


def custom_distance(file):
    data = {}
    with open(file, "r") as infile:
        for l in infile:
            labelA, labelB, dist = l.strip().split("\t")
            labelA = frozenset([labelA])
            labelB = frozenset([labelB])
            data[frozenset([labelA, labelB])] = float(dist)
    return lambda x, y: data[frozenset([x, y])]


def jaro_similarity(s1, s2):
    """
   Computes the Jaro similarity between 2 sequences from:

        Matthew A. Jaro (1989). Advances in record linkage methodology
        as applied to the 1985 census of Tampa Florida. Journal of the
        American Statistical Association. 84 (406): 414-20.

    The Jaro distance between is the min no. of single-character transpositions
    required to change one word into another. The Jaro similarity formula from
    https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance :

        jaro_sim = 0 if m = 0 else 1/3 * (m/|s_1| + m/s_2 + (m-t)/m)

    where:
        - |s_i| is the length of string s_i
        - m is the no. of matching characters
        - t is the half no. of possible transpositions.

    """
    # First, store the length of the strings
    # because they will be re-used several times.
    len_s1, len_s2 = len(s1), len(s2)

    # The upper bound of the distance for being a matched character.
    match_bound = max(len_s1, len_s2) // 2 - 1

    # Initialize the counts for matches and transpositions.
    matches = 0  # no.of matched characters in s1 and s2
    transpositions = 0  # no. of transpositions between s1 and s2
    flagged_1 = []  # positions in s1 which are matches to some character in s2
    flagged_2 = []  # positions in s2 which are matches to some character in s1

    # Iterate through sequences, check for matches and compute transpositions.
    for i in range(len_s1):  # Iterate through each character.
        upperbound = min(i + match_bound, len_s2 - 1)
        lowerbound = max(0, i - match_bound)
        for j in range(lowerbound, upperbound + 1):
            if s1[i] == s2[j] and j not in flagged_2:
                matches += 1
                flagged_1.append(i)
                flagged_2.append(j)
                break
    flagged_2.sort()
    for i, j in zip(flagged_1, flagged_2):
        if s1[i] != s2[j]:
            transpositions += 1

    if matches == 0:
        return 0
    else:
        return (
            1
            / 3
            * (
                matches / len_s1
                + matches / len_s2
                + (matches - transpositions // 2) / matches
            )
        )


def jaro_winkler_similarity(s1, s2, p=0.1, max_l=4):
    """
    The Jaro Winkler distance is an extension of the Jaro similarity in:

        William E. Winkler. 1990. String Comparator Metrics and Enhanced
        Decision Rules in the Fellegi-Sunter Model of Record Linkage.
        Proceedings of the Section on Survey Research Methods.
        American Statistical Association: 354-359.
    such that:

        jaro_winkler_sim = jaro_sim + ( l * p * (1 - jaro_sim) )

    where,

        - jaro_sim is the output from the Jaro Similarity,
        see jaro_similarity()
        - l is the length of common prefix at the start of the string
            - this implementation provides an upperbound for the l value
              to keep the prefixes.A common value of this upperbound is 4.
        - p is the constant scaling factor to overweigh common prefixes.
          The Jaro-Winkler similarity will fall within the [0, 1] bound,
          given that max(p)<=0.25 , default is p=0.1 in Winkler (1990)


    Test using outputs from https://www.census.gov/srd/papers/pdf/rr93-8.pdf
    from "Table 5 Comparison of String Comparators Rescaled between 0 and 1"

    >>> winkler_examples = [("billy", "billy"), ("billy", "bill"), ("billy", "blily"),
    ... ("massie", "massey"), ("yvette", "yevett"), ("billy", "bolly"), ("dwayne", "duane"),
    ... ("dixon", "dickson"), ("billy", "susan")]

    >>> winkler_scores = [1.000, 0.967, 0.947, 0.944, 0.911, 0.893, 0.858, 0.853, 0.000]
    >>> jaro_scores =    [1.000, 0.933, 0.933, 0.889, 0.889, 0.867, 0.822, 0.790, 0.000]

        # One way to match the values on the Winkler's paper is to provide a different
    # p scaling factor for different pairs of strings, e.g.
    >>> p_factors = [0.1, 0.125, 0.20, 0.125, 0.20, 0.20, 0.20, 0.15, 0.1]

    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore


    Test using outputs from https://www.census.gov/srd/papers/pdf/rr94-5.pdf from
    "Table 2.1. Comparison of String Comparators Using Last Names, First Names, and Street Names"

    >>> winkler_examples = [('SHACKLEFORD', 'SHACKELFORD'), ('DUNNINGHAM', 'CUNNIGHAM'),
    ... ('NICHLESON', 'NICHULSON'), ('JONES', 'JOHNSON'), ('MASSEY', 'MASSIE'),
    ... ('ABROMS', 'ABRAMS'), ('HARDIN', 'MARTINEZ'), ('ITMAN', 'SMITH'),
    ... ('JERALDINE', 'GERALDINE'), ('MARHTA', 'MARTHA'), ('MICHELLE', 'MICHAEL'),
    ... ('JULIES', 'JULIUS'), ('TANYA', 'TONYA'), ('DWAYNE', 'DUANE'), ('SEAN', 'SUSAN'),
    ... ('JON', 'JOHN'), ('JON', 'JAN'), ('BROOKHAVEN', 'BRROKHAVEN'),
    ... ('BROOK HALLOW', 'BROOK HLLW'), ('DECATUR', 'DECATIR'), ('FITZRUREITER', 'FITZENREITER'),
    ... ('HIGBEE', 'HIGHEE'), ('HIGBEE', 'HIGVEE'), ('LACURA', 'LOCURA'), ('IOWA', 'IONA'), ('1ST', 'IST')]

    >>> jaro_scores =   [0.970, 0.896, 0.926, 0.790, 0.889, 0.889, 0.722, 0.467, 0.926,
    ... 0.944, 0.869, 0.889, 0.867, 0.822, 0.783, 0.917, 0.000, 0.933, 0.944, 0.905,
    ... 0.856, 0.889, 0.889, 0.889, 0.833, 0.000]

    >>> winkler_scores = [0.982, 0.896, 0.956, 0.832, 0.944, 0.922, 0.722, 0.467, 0.926,
    ... 0.961, 0.921, 0.933, 0.880, 0.858, 0.805, 0.933, 0.000, 0.947, 0.967, 0.943,
    ... 0.913, 0.922, 0.922, 0.900, 0.867, 0.000]

        # One way to match the values on the Winkler's paper is to provide a different
    # p scaling factor for different pairs of strings, e.g.
    >>> p_factors = [0.1, 0.1, 0.1, 0.1, 0.125, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.20,
    ... 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


    >>> for (s1, s2), jscore, wscore, p in zip(winkler_examples, jaro_scores, winkler_scores, p_factors):
    ...     if (s1, s2) in [('JON', 'JAN'), ('1ST', 'IST')]:
    ...         continue  # Skip bad examples from the paper.
    ...     assert round(jaro_similarity(s1, s2), 3) == jscore
    ...     assert round(jaro_winkler_similarity(s1, s2, p=p), 3) == wscore



    This test-case proves that the output of Jaro-Winkler similarity depends on
    the product  l * p and not on the product max_l * p. Here the product max_l * p > 1
    however the product l * p <= 1

    >>> round(jaro_winkler_similarity('TANYA', 'TONYA', p=0.1, max_l=100), 3)
    0.88


    """
    # To ensure that the output of the Jaro-Winkler's similarity
    # falls between [0,1], the product of l * p needs to be
    # also fall between [0,1].
    if not 0 <= max_l * p <= 1:
        warnings.warn(
            str(
                "The product  `max_l * p` might not fall between [0,1]."
                "Jaro-Winkler similarity might not be between 0 and 1."
            )
        )

    # Compute the Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)

    # Initialize the upper bound for the no. of prefixes.
    # if user did not pre-define the upperbound,
    # use shorter length between s1 and s2

    # Compute the prefix matches.
    l = 0
    # zip() will automatically loop until the end of shorter string.
    for s1_i, s2_i in zip(s1, s2):
        if s1_i == s2_i:
            l += 1
        else:
            break
        if l == max_l:
            break
    # Return the similarity value as described in docstring.
    return jaro_sim + (l * p * (1 - jaro_sim))


def demo():
    string_distance_examples = [
        ("rain", "shine"),
        ("abcdef", "acbdef"),
        ("language", "lnaguaeg"),
        ("language", "lnaugage"),
        ("language", "lngauage"),
    ]
    for s1, s2 in string_distance_examples:
        print("Edit distance btwn '%s' and '%s':" % (s1, s2), edit_distance(s1, s2))
        print(
            "Edit dist with transpositions btwn '%s' and '%s':" % (s1, s2),
            edit_distance(s1, s2, transpositions=True),
        )
        print("Jaro similarity btwn '%s' and '%s':" % (s1, s2), jaro_similarity(s1, s2))
        print(
            "Jaro-Winkler similarity btwn '%s' and '%s':" % (s1, s2),
            jaro_winkler_similarity(s1, s2),
        )
        print(
            "Jaro-Winkler distance btwn '%s' and '%s':" % (s1, s2),
            1 - jaro_winkler_similarity(s1, s2),
        )
    # s1 = set([1, 2, 3, 4])
    # s2 = set([3, 4, 5])
    # print("s1:", s1)
    # print("s2:", s2)
    # print("Binary distance:", binary_distance(s1, s2))
    # print("Jaccard distance:", jaccard_distance(s1, s2))
    # print("MASI distance:", masi_distance(s1, s2))

def demo1():
    string_distance_examples = [
        ("rainy", "shine"),
        ("abcdef", "acbdef"),
        ("language", "lnaguaeg"),
        ("language", "lnaugage"),
        ("language", "lngauage"),
    ]
    for s1, s2 in string_distance_examples:
        print("Levenshtein distance btwn '%s' and '%s':" % (s1, s2), LevenshteinEditDistance().compute(s1, s2))
        print("Damerau-Levenshtein distance btwn '%s' and '%s':" % (s1, s2), DamerauLevenshteinDistance().compute(s1, s2))
        print("Hamming distance btwn '%s' and '%s':" % (s1, s2), HammingDistance().compute(s1, s2))
if __name__ == "__main__":
    demo1()
