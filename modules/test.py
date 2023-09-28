with open('scene_samples.txt', 'r') as file:
    # Read lines from the file and convert them to integers
    numbers = [int(line.strip()) for line in file]

# Now, the 'numbers' list contains the numbers from the file
print(numbers)