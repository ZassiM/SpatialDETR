# Initialize empty lists to store the numbers and sentences
numbers = []
sentences = []

# Open the file for reading
with open('scenes.txt', 'r') as file:
    # Read each line in the file
    for line in file:
        # Split the line by the '|' character
        parts = line.split('|')
        
        # Check if there are at least two parts (a number and a sentence)
        if len(parts) >= 2:
            # Extract and clean the number (assuming it's at the beginning of the first part)
            number = int(parts[0].strip().split()[-1])
            
            # Extract and clean the sentence (after the '|' character)
            sentence = parts[1].strip()
            
            # Append the number and sentence to their respective lists
            numbers.append(number)
            sentences.append(sentence)

# Print the extracted numbers and sentences
print("Numbers:", numbers)
print("Sentences:", sentences)
