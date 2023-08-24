"""
string2string similarity
src = https://github.com/stanfordnlp/string2string


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

from typing import List, Union, Tuple
import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np

# for dev purposes
import sys
# sys.path.append("/Users/dowon/nltk_ko/nltk/misc")
from nltk.misc.string2string_word_embeddings import GloVeEmbeddings
# from string2string_word_embeddings import GloVeEmbeddings


# Cosine similarity class
class CosineSimilarity:
    def __init__(self) -> None:
        r"""
        This function initializes the CosineSimilarity class.
        """
        pass


    # Compute (tensor)
    def _compute_tensor(self,
        x1: Tensor,
        x2: Tensor,
        dim: int = 1,
        eps: float = 1e-8
    ) -> Tensor:
        r"""
        Computes the cosine similarity between two tensors along a given dimension.

        Arguments:
            x1 (Tensor): First tensor.
            x2 (Tensor): Second tensor.
            dim (int): Dimension to compute cosine similarity.
            eps (float): Epsilon value.

        Returns:
            Tensor: Cosine similarity between two tensors along a given dimension.
        """
        # Make sure that x1 and x2 are float tensors
        if x1.dtype != torch.float:
            x1 = x1.float()
        if x2.dtype != torch.float:
            x2 = x2.float()
        # Compute cosine similarity between two tensors
        return F.cosine_similarity(x1, x2, dim, eps)
    

    # Compute (numpy)
    def _compute_numpy(self,
        x1: np.ndarray,
        x2: np.ndarray,
        dim: int = 1,
        eps: float = 1e-8
    ) -> np.ndarray:
        r"""
        Computes the cosine similarity between two numpy arrays along a given dimension.

        Arguments:
            x1 (np.ndarray): First numpy array.
            x2 (np.ndarray): Second numpy array.
            dim (int): Dimension (or axis in the numpy realm) to compute cosine similarity.
            eps (float): Epsilon value (to prevent division by zero).

        Returns:
            np.ndarray: Cosine similarity between two numpy arrays along a given dimension.
        """
        # Compute cosine similarity between two numpy arrays along a given dimension "dim"
        return np.sum(x1 * x2, axis=dim) / np.maximum(np.linalg.norm(x1, axis=dim) * np.linalg.norm(x2, axis=dim), eps)


    # Compute
    def compute(self,
        x1: Union[Tensor, np.ndarray],
        x2: Union[Tensor, np.ndarray],
        dim: int = 0,
        eps: float = 1e-8
    ) -> Union[Tensor, np.ndarray]:
        r"""
        Computes the cosine similarity between two tensors (or numpy arrays) along a given dimension.
        
        * For two (non-zero) vectors, :math:`x_1` and :math:`x_2`, the cosine similarity is defined as follows:

            .. math::
                :nowrap:

                \begin{align}
                    \texttt{cosine-similarity}(x_1, x_2) & = |x_1|| \ ||x_2|| \cos(\theta) \\
                    & = \frac{x_1 \cdot x_2}{||x_1|| \ ||x_2||} \\
                    & = \frac{\sum_{i=1}^n x_{1i} x_{2i}}{\sqrt{\sum_{i=1}^n x_{1i}^2} \sqrt{\sum_{i=1}^n x_{2i}^2}}
                \end{align}
        
            where :math:`\theta` denotes the angle between the vectors, :math:`\cdot` the dot product, and :math:`||\cdot||` the norm operator.
                
        * In practice, the cosine similarity is computed as follows:

            .. math::
                :nowrap:

                \begin{align}
                    \texttt{cosine-similarity}(x_1, x_2) & = \frac{x_1 \cdot x_2}{\max(||x_1|| ||x_2||, \epsilon)}
                \end{align}
                
            where :math:`\epsilon` is a small value to avoid division by zero.


        Arguments:
            x1 (Union[Tensor, np.ndarray]): First tensor (or numpy array).
            x2 (Union[Tensor, np.ndarray]): Second tensor (or numpy array).
            dim (int): Dimension to compute cosine similarity (default: 0).
            eps (float): Epsilon value (to avoid division by zero).

        Returns:
            Union[Tensor, np.ndarray]: Cosine similarity between two tensors (or numpy arrays) along a given dimension.

        Raises:
            TypeError: If x1 and x2 are not of the same type (either tensor or numpy array).
            TypeError: If x1 and x2 are not tensors or numpy arrays.
        """
        # Check if x1 and x2 are of the same type (either tensor or numpy array)
        if type(x1) != type(x2):
            raise TypeError("x1 and x2 must be of the same type (either tensor or numpy array).")
        
        # If x1 and x2 are tensors
        if type(x1) == Tensor:
            # Compute cosine similarity
            return self._compute_tensor(x1, x2, dim, eps)
        # If x1 and x2 are numpy arrays
        elif type(x1) == np.ndarray:
            # Compute cosine similarity
            return self._compute_numpy(x1, x2, dim, eps)
        # If x1 and x2 are not tensors or numpy arrays
        else:
            raise TypeError("x1 and x2 must be either tensors or numpy arrays.")

def demo():
    array1 = np.array([20, 65, 1])
    array2 = np.array([98, 67, 548])

    print("demo : ", CosineSimilarity().compute(array1, array2))

if __name__ == "__main__":
    demo()