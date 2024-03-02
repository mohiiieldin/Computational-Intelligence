import time

def generate_all_binary_numbers(length):
    """
    Generates all possible binary numbers of the given length.
    """
    num_numbers = 2 ** length
    all_numbers = []

    for i in range(num_numbers):
        binary_number = []
        for j in range(length):
            bit = (i >> j) & 1
            binary_number.append(bit)
        all_numbers.append(binary_number)
    return all_numbers

def find_best_binary_number(length):
    """
    Finds the binary number with the maximum sum of ones (1s).
    """
    all_numbers = generate_all_binary_numbers(length)
    max_ones = -1
    best_number = []

    for binary_number in all_numbers:
        num_ones = sum(binary_number)  # Calculate sum of ones

        if num_ones > max_ones:
            max_ones = num_ones
            best_number = binary_number

    return best_number

# Example usage:

start = time.time()
result = find_best_binary_number(23)
brute_force_time = time.time() - start
print("Brute Force Time For Length 23:",round(brute_force_time,1),'Seconds')
print(f"The binary number with the most 1s: {result}")
