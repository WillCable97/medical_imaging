import numpy as np

def select_evenly_spaced_entries(arr, num_to_select):
    # Get the size of the last dimension
    last_dim_size = arr.shape[-1]

    # Calculate the spacing based on the desired number to select
    spacing = max(last_dim_size // num_to_select, 1)

    # Create an array of indices to select
    indices = np.arange(0, last_dim_size, spacing)[:num_to_select]

    # Select elements using the calculated indices
    arr[..., -1] = arr[..., -1, indices]

# Example usage:
nested_array = np.random.rand(10, 20, 256, 38)  # Example 4D array

# Choose the desired number of elements to select
desired_num_to_select = 256

select_evenly_spaced_entries(nested_array, desired_num_to_select)

# Output the modified nested array
print(nested_array)
