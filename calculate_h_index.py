def calculate_h_index(citations):
    """
    Calculate the H-index from a list of citations.
    
    Args:
        citations (list): A list of integers representing citation counts for each paper.
    
    Returns:
        int: The H-index of the researcher.
    """
    citations.sort(reverse=True)  # Sort citations in descending order
    h_index = 0
    for i, citation in enumerate(citations):
        if citation >= i + 1:
            h_index = i + 1
        else:
            break
    return h_index

# Example usage
if __name__ == "__main__":
    # Example citation list
    citations = [3, 0, 6, 1, 5]
    h_index = calculate_h_index(citations)
    print(f"H-index: {h_index}")