from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def load_and_center_dataset(filename):
    """
    Load dataset from .npy file and center it by subtracting the mean.

    Args:
        filename (str): Path to the .npy file

    Returns:
        numpy.ndarray: Centered dataset (n x d matrix)
    """
    x = np.load(filename)
    centered_x = x - np.mean(x, axis = 0)

    return centered_x

def get_covariance(dataset):
    """
    Calculate the sample covariance matrix of the dataset.

    Args:
        dataset (numpy.ndarray): Centered dataset (n x d matrix)

    Returns:
        numpy.ndarray: Covariance matrix (d x d matrix)
    """
    n = dataset.shape[0]
    S = (1/(n-1)) * np.dot(dataset.T, dataset)
    return S

def get_eig(S, k):
    """
    Get the k largest eigenvalues and corresponding eigenvectors.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        k (int): Number of largest eigenvalues/eigenvectors to return

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    d = S.shape[0]
    eigenvalues, eigenvectors = eigh(S, subset_by_index=[d - k, d - 1])
    sorted_eigenvalues = np.flip(eigenvalues)
    sorted_eigenvectors = np.flip(eigenvectors, axis=1)
    Lambda = np.diag(sorted_eigenvalues)

    return (Lambda, sorted_eigenvectors)

def get_eig_prop(S, prop):
    """
    Get eigenvalues and eigenvectors that explain more than prop proportion of variance.

    Args:
        S (numpy.ndarray): Covariance matrix (d x d)
        prop (float): Minimum proportion of variance to explain (0 <= prop <= 1)

    Returns:
        tuple: (Lambda, U) where Lambda is diagonal matrix of eigenvalues
               and U is matrix of corresponding eigenvectors as columns
    """
    total_variance = np.trace(S)
    if total_variance == 0:
        return (np.array([[]]), np.array([[]]))
    
    all_eigenvalues, all_eigenvectors = eigh(S)

    desc_eigenvalues = np.flip(all_eigenvalues)
    desc_eigenvectors = np.flip(all_eigenvectors, axis=1)

    proportions = desc_eigenvalues / total_variance
    indices_to_keep = proportions > prop

    selected_eigenvalues = desc_eigenvalues[indices_to_keep]
    selected_eigenvectors = desc_eigenvectors[:, indices_to_keep]
    Lambda = np.diag(selected_eigenvalues)

    return Lambda, selected_eigenvalues

def project_and_reconstruct_image(image, U):
    """
    Project image to PCA subspace and reconstruct it back to original dimension.

    Args:
        image (numpy.ndarray): Flattened image vector (d x 1)
        U (numpy.ndarray): Matrix of eigenvectors (d x m)

    Returns:
        numpy.ndarray: Reconstructed image as flattened d x 1 vector
    """
    alpha = np.dot(U.T, image)
    reconstructed_image = np.dot(U, alpha)
    return reconstructed_image

def display_image(im_orig_fullres, im_orig, im_reconstructed):
    """
    Display three images side by side: original high-res, original, and reconstructed.

    Args:
        im_orig_fullres (numpy.ndarray): Original high-resolution image
        im_orig (numpy.ndarray): Original low-resolution image
        im_reconstructed (numpy.ndarray): Reconstructed image from PCA

    Returns:
        tuple: (fig, ax1, ax2, ax3) matplotlib figure and axes objects
    """

    # Please use the format below to ensure grading consistency
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(9,3), ncols=3)
    fig.tight_layout()

    # Reshape all the image vectors into 2D or 3D matrices
    reshaped_fullres = im_orig_fullres.reshape(218, 178, 3)
    reshaped_orig = im_orig.reshape(60, 50)
    reshaped_reconstructed = im_reconstructed.reshape(60, 50)

    # --- Plot 1: Original High Res ---
    ax1.imshow(reshaped_fullres, aspect='equal')
    ax1.set_title("Original High Res")

    # --- Plot 2: Original Low Res (This part was missing) ---
    im2 = ax2.imshow(reshaped_orig, aspect='equal', cmap='gray') 
    ax2.set_title("Original")
    fig.colorbar(im2, ax=ax2)

    # --- Plot 3: Reconstructed Image ---
    im3 = ax3.imshow(reshaped_reconstructed, aspect='equal', cmap='gray') 
    ax3.set_title("Reconstructed")
    fig.colorbar(im3, ax=ax3)

    return fig, ax1, ax2, ax3


# =============================================================================
# N-GRAM LANGUAGE MODEL FUNCTIONS
# =============================================================================

class NGramCharLM:
    """
    N-gram Character Language Model
    
    This class implements an n-gram character-level language model.
    Students should implement the missing methods.
    """
    
    def __init__(self, n=5):
        """
        Initialize the n-gram language model.
        
        Args:
            n (int): The order of the n-gram model (must be >= 1)
        """
        assert n >= 1, "n must be at least 1"
        self.n = n
        self.counts = {}  # dict[str, dict[str, int]] - context -> {char: count}
        self.vocab = set()  # set of all characters seen during training
        self.trained = False
    
    def fit(self, text: str):
        """
        Train the n-gram model on the given text.
        
        Args:
            text (str): Training text
            
        Returns:
            NGramCharLM: self (for method chaining)


        Examples:
        If n=3 and text="banana", then counts should look like:
        {
            ""   : {"b": 1},         # at i=0, ctx="", ch="b"
            "b"  : {"a": 1},         # at i=1, ctx="b", ch="a"
            "ba" : {"n": 1},         # at i=2, ctx="ba", ch="n"
            "an" : {"a": 2},         # at i=3,5, ctx="an", ch="a"
            "na" : {"n": 1}          # at i=4, ctx="na", ch="n"
        }

        If n=2 and text="abab", then counts should look like:
        {
            ""  : {"a": 1},          # at i=0, ctx="", ch="a"
            "a" : {"b": 2},          # at i=1,3, ctx="a", ch="b"
            "b" : {"a": 1}           # at i=2, ctx="b", ch="a"
        }

        Note: Context is always the last (n-1) characters before current position.
          For positions near the beginning, context may be shorter than (n-1).
        """
        # Your implementation goes here!
        raise NotImplementedError
    
    def _probs_for_context(self, context: str):
        """
        Get probability distribution over next characters for a given context.
        
        Args:
            context (str): The context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities
        """
        # Your implementation goes here!
        raise NotImplementedError
    
    def prob(self, s: str) -> float:
        """
        Calculate the probability of a string.
        
        Args:
            s (str): String to calculate probability for
            
        Returns:
            float: Probability of the string
        """
        return math.exp(self.logprob(s))
    
    def logprob(self, s: str) -> float:
        """
        Calculate the log-probability of a string.
        
        Args:
            s (str): String to calculate log-probability for
            
        Returns:
            float: Log-probability of the string
        """
        lp = 0.0
        for i in range(len(s)):
            # Get context for character at position i
            ctx = s[max(0, i - (self.n - 1)):i]
            
            # Get probability of character given context
            p_next = self._probs_for_context(ctx).get(s[i], 0.0)
            
            if p_next <= 0.0:
                return float("-inf")
            
            lp += math.log(p_next)
        
        return lp
    
    def next_char_distribution(self, context: str):
        """
        Get the probability distribution over next characters for given context.
        
        Args:
            context (str): Context string
            
        Returns:
            dict[str, float]: Dictionary mapping characters to probabilities,
                            sorted by probability in descending order
        """
        d = self._probs_for_context(context)
        return dict(sorted(d.items(), key=lambda kv: -kv[1]))

    def generate(self, num_chars: int, seed: str = "") -> str:
        """
        Generate text using the trained model.

        Args:
            num_chars (int): Number of characters to generate
            seed (str): Initial string to start generation

        Returns:
            str: Generated text (seed + num_chars new characters)
        """
        out = list(seed)

        for _ in range(num_chars):
            # Get probability distribution for current context
            dist = self._probs_for_context("".join(out))

            if not dist:
                break

            # Extract characters and probabilities
            chars, probs = zip(*dist.items())

            # Cumulative sampling
            r = random.random()
            s = 0.0
            pick = chars[-1]  # fallback to last character

            for ch, p in zip(chars, probs):
                s += p
                if r <= s:
                    pick = ch
                    break

            out.append(pick)

        return "".join(out)

if __name__ == "__main__":
    X = load_and_center_dataset('celeba_60x50.npy')

    S = get_covariance(X)
    
    Lambda, U = get_eig(S, 50)
    
    celeb_idx = 34 
    
    x = X[celeb_idx]
    
    x_fullres = np.load('celeba_218x178x3.npy')[celeb_idx]
    
    reconstructed = project_and_reconstruct_image(x, U)
    
    fig, ax1, ax2, ax3 = display_image(x_fullres, x, reconstructed)
    
    plt.show()