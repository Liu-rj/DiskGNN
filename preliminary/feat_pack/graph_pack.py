from itertools import combinations

def calculate_pairwise_overlap(sequences):
    """
    Calculates pairwise overlap of sequences and returns a matrix with the overlaps.
    """
    n = len(sequences)
    pairwise_overlap = [[set() for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            overlap = set(sequences[i]).intersection(sequences[j])
            pairwise_overlap[i][j] = overlap
            pairwise_overlap[j][i] = overlap  # Symmetric matrix

    return pairwise_overlap

def remove_triplet_overlap_from_pairwise(sequences, pairwise_overlap):
    """
    Removes triplet overlap elements from pairwise overlaps.
    """
    for triplet in combinations(range(len(sequences)), 3):
        overlap = set(sequences[triplet[0]]).intersection(sequences[triplet[1]], sequences[triplet[2]])
        for pair in combinations(triplet, 2):
            pairwise_overlap[pair[0]][pair[1]] -= overlap
            pairwise_overlap[pair[1]][pair[0]] -= overlap

    return pairwise_overlap

# Example usage
sequences = [[1, 2, 3], [3, 4, 5], [1, 5, 6], [2, 3, 7]]
pairwise_overlap = calculate_pairwise_overlap(sequences)
pairwise_overlap = remove_triplet_overlap_from_pairwise(sequences, pairwise_overlap)

print(pairwise_overlap)
